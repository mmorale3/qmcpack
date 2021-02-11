//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2020 QMCPACK developers.
//
// File developed by: Miguel Morales, moralessilva2@llnl.gov, Lawrence Livermore National Laboratory
//                    Ken Esler, kpesler@gmail.com, University of Illinois at Urbana-Champaign
//                    Jeremy McMinnis, jmcminis@gmail.com, University of Illinois at Urbana-Champaign
//                    Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//                    Mark A. Berrill, berrillma@ornl.gov, Oak Ridge National Laboratory
//                    Mark Dewing, mdewing@anl.gov, Argonne National Laboratory
//
// File created by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//////////////////////////////////////////////////////////////////////////////////////


#include "QMCOptimizeBatched.h"
#include "Particle/HDFWalkerIO.h"
#include "OhmmsData/AttributeSet.h"
#include "Message/CommOperators.h"
#include "Optimize/CGOptimization.h"
#include "Optimize/testDerivOptimization.h"
#include "Optimize/DampedDynamics.h"
#include "QMCDrivers/VMC/VMCBatched.h"
#include "QMCDrivers/WFOpt/QMCCostFunctionBatched.h"
#include "QMCHamiltonians/HamiltonianPool.h"

namespace qmcplusplus
{
QMCOptimizeBatched::QMCOptimizeBatched(const ProjectData& project_data,
                                       MCWalkerConfiguration& w,
                                       TrialWaveFunction& psi,
                                       QMCHamiltonian& h,
                                       QMCDriverInput&& qmcdriver_input,
                                       VMCDriverInput&& vmcdriver_input,
                                       MCPopulation&& population,
                                       SampleStack& samples,
                                       Communicate* comm)
    : QMCDriverNew(project_data,
                   std::move(qmcdriver_input),
                   std::move(population),
                   psi,
                   h,
                   "QMCOptimizeBatched::",
                   comm,
                   "QMCOptimizeBatched"),
      PartID(0),
      NumParts(1),
      optSolver(0),
      vmcEngine(0),
      wfNode(NULL),
      optNode(NULL),
      vmcdriver_input_(vmcdriver_input),
      samples_(samples),
      W(w)
{
  optmethod = "test";
}

/** Clean up the vector */
QMCOptimizeBatched::~QMCOptimizeBatched()
{
  delete vmcEngine;
  delete optSolver;
}

/** Add configuration files for the optimization
 * @param a root of a hdf5 configuration file
 */
void QMCOptimizeBatched::addConfiguration(const std::string& a)
{
  if (a.size())
    ConfigFile.push_back(a);
}

/** Reimplement QMCDriver::run
 */
bool QMCOptimizeBatched::run()
{
  //close files automatically generated by QMCDriver
  //branchEngine->finalize();
  //generate samples
  generateSamples();

  app_log() << "<opt stage=\"setup\">" << std::endl;
  app_log() << "  <log>" << std::endl;
  //reset the rootname
  optTarget->setRootName(get_root_name());
  optTarget->setWaveFunctionNode(wfNode);
  optTarget->setRng(vmcEngine->getRng());
  app_log() << "   Reading configurations from h5FileRoot " << std::endl;
  //get configuration from the previous run
  Timer t1;
  optTarget->getConfigurations("");
  optTarget->checkConfigurations();
  app_log() << "  Execution time = " << std::setprecision(4) << t1.elapsed() << std::endl;
  app_log() << "  </log>" << std::endl;
  app_log() << "</opt>" << std::endl;
  app_log() << "<opt stage=\"main\" walkers=\"" << optTarget->getNumSamples() << "\">" << std::endl;
  app_log() << "  <log>" << std::endl;
  // FIXME: Ye to Mark: branch_engine_ of QMCOptimizeBatched doesn't hold anything.
  // Hopefully this was not affecting anything.
  //optTarget->setTargetEnergy(branch_engine_->getEref());
  t1.restart();
  bool success = optSolver->optimize(optTarget.get());
  app_log() << "  Execution time = " << std::setprecision(4) << t1.elapsed() << std::endl;
  ;
  app_log() << "  </log>" << std::endl;
  optTarget->reportParameters();

  app_log() << "</opt>" << std::endl;
  app_log() << "</optimization-report>" << std::endl;

  return (optTarget->getReportCounter() > 0);
}

void QMCOptimizeBatched::generateSamples()
{
  Timer t1;
  app_log() << "<optimization-report>" << std::endl;

  t1.restart();

  samples_.resetSampleCount();
  population_.set_variational_parameters(optTarget->getOptVariables());

  vmcEngine->run();
  app_log() << "  Execution time = " << std::setprecision(4) << t1.elapsed() << std::endl;
  app_log() << "</vmc>" << std::endl;

  h5_file_root_ = get_root_name();
}

/** Parses the xml input file for parameter definitions for the wavefunction optimization.
 * @param q current xmlNode
 * @return true if successful
 */
void QMCOptimizeBatched::process(xmlNodePtr q)
{
  std::string vmcMove("pbyp");
  std::string useGPU("no");
  OhmmsAttributeSet oAttrib;
  oAttrib.add(vmcMove, "move");
  oAttrib.add(useGPU, "gpu");
  oAttrib.put(q);
  xmlNodePtr qsave = q;
  xmlNodePtr cur   = qsave->children;
  int pid          = OHMMS::Controller->rank();

  int crowd_size     = 1;
  int num_opt_crowds = 1;
  ParameterSet param_set;
  param_set.add(crowd_size, "opt_crowd_size");
  param_set.add(num_opt_crowds, "opt_num_crowds");
  param_set.put(q);

  while (cur != NULL)
  {
    std::string cname((const char*)(cur->name));
    if (cname == "mcwalkerset")
    {
      mcwalkerNodePtr.push_back(cur);
    }
    else if (cname.find("optimize") < cname.size())
    {
      const XMLAttrString att(cur, "method");
      if (!att.empty())
        optmethod = att;
      optNode = cur;
    }
    cur = cur->next;
  }
  //no walkers exist, add 10
  //if (W.getActiveWalkers() == 0)
  //  addWalkers(omp_get_max_threads());
  //NumOfVMCWalkers = W.getActiveWalkers();
  //create VMC engine
  if (vmcEngine == 0)
  {
    QMCDriverInput qmcdriver_input_copy = qmcdriver_input_;
    VMCDriverInput vmcdriver_input_copy = vmcdriver_input_;
    vmcEngine = new VMCBatched(project_data_, std::move(qmcdriver_input_copy), std::move(vmcdriver_input_copy),
                               MCPopulation(myComm->size(), myComm->rank(), population_.getWalkerConfigsRef(),
                                            population_.get_golden_electrons(), &Psi, &H),
                               Psi, H, samples_, myComm);

    vmcEngine->setUpdateMode(vmcMove[0] == 'p');
    bool AppendRun = false;
    vmcEngine->setStatus(get_root_name(), h5_file_root_, AppendRun);
    vmcEngine->process(qsave);
    vmcEngine->enable_sample_collection();
  }
  if (optSolver == 0)
  {
    if (optmethod == "anneal")
    {
      app_log() << " Annealing optimization using DampedDynamics" << std::endl;
      optSolver = new DampedDynamics<RealType>;
    }
    else if (optmethod == "test")
    {
      app_log() << "Test and output parameter derivatives (batched): " << std::endl;
      optSolver = new testDerivOptimization<RealType>(get_root_name());
    }
    else
    {
      app_log() << "Unknown optimize method: " << optmethod << std::endl;
      APP_ABORT("QMCOptimizeBatched::process");
    } //set the stream
    optSolver->setOstream(&app_log());
  }
  if (optNode == NULL)
    optSolver->put(qsave);
  else
    optSolver->put(optNode);
  bool success = true;
  //allways reset optTarget
  optTarget = std::make_unique<QMCCostFunctionBatched>(W, Psi, H, samples_, num_opt_crowds, crowd_size, myComm);
  optTarget->setStream(&app_log());
  success = optTarget->put(q);

  // This code is also called when setting up vmcEngine.  Would be nice to not duplicate the call.
  QMCDriverNew::AdjustedWalkerCounts awc =
      adjustGlobalWalkerCount(myComm->size(), myComm->rank(), qmcdriver_input_.get_total_walkers(),
                              qmcdriver_input_.get_walkers_per_rank(), 1.0, qmcdriver_input_.get_num_crowds());
  QMCDriverNew::startup(q, awc);
}
} // namespace qmcplusplus
