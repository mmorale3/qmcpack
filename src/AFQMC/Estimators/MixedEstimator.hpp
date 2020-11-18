//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: Miguel Morales, moralessilva2@llnl.gov, Lawrence Livermore National Laboratory
//
// File created by: Miguel Morales, moralessilva2@llnl.gov, Lawrence Livermore National Laboratory
//////////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_AFQMC_MIXED_ESTIMATOR_HPP
#define QMCPLUSPLUS_AFQMC_MIXED_ESTIMATOR_HPP

#include "AFQMC/config.h"
#include <vector>
#include <string>
#include <iostream>

#include "hdf/hdf_multi.h"
#include "hdf/hdf_archive.h"
#include "OhmmsData/libxmldefs.h"
#include "Utilities/Timer.h"

#include "AFQMC/Estimators/MixedObsHandler.hpp"
#include "AFQMC/Wavefunctions/Wavefunction.hpp"
#include "AFQMC/Walkers/WalkerSet.hpp"

namespace qmcplusplus
{
namespace afqmc
{
/*
 * Top class for mixed estimators. 
 * An instance of this class will manage a set of observables evaluated at the mixed distribution.
 */
class MixedEstimator : public EstimatorBase
{

public:
  MixedEstimator(afqmc::TaskGroup_& tg_,
                          AFQMCInfo& info,
                          std::string name,
                          xmlNodePtr cur,
                          WALKER_TYPES wlk,
                          WalkerSet& wset,
                          Wavefunction& wfn)
      : EstimatorBase(info),
        TG(tg_),
        walker_type(wlk),
        observ0(TG, info, name, cur, wlk, wfn),
        wfn0(wfn)
  {
    if (cur != NULL)
    {
      ParameterSet m_param;
      m_param.add(block_size, "block_size", "int");
      m_param.add(nblocks_skip, "nskip", "int");
      m_param.put(cur);
    }
    writer = (TG.getGlobalRank() == 0);
  }

  ~MixedEstimator() {}

  void accumulate_step(WalkerSet& wset, std::vector<ComplexType>& curData) {}

  void accumulate_block(WalkerSet& wset)
  {
    accumulated_in_last_block = false;

    // 0. skip if requested
    // MAM: problematic on restarts!!!
    if (iblock < nblocks_skip) { 
      iblock++;
      return;
    }

    AFQMCTimers[mixed_estimator_timer]->start();
    observ0.accumulate(wset);
    iblock++;
    accumulated_in_last_block = true;
    AFQMCTimers[mixed_estimator_timer]->stop();
  }

  void tags(std::ofstream& out)
  {
    if (writer)
      out << "MixedEstim_timer ";
  }

  void print(std::ofstream& out, hdf_archive& dump, WalkerSet& wset)
  {
    // I doubt we will ever collect a billion blocks of data.
    if (writer)
    {
      out << std::setprecision(5) << AFQMCTimers[mixed_estimator_timer]->get_total() << " ";
      AFQMCTimers[mixed_estimator_timer]->reset();
    }
    if (accumulated_in_last_block)
    {
      if (writer)
      {
        dump.push("Observables");
        dump.push("Mixed");
      }
      observ0.print(iblock, dump);
      if (writer)
      {
        dump.pop();
        dump.pop();
      }
    }
  }

private:
  TaskGroup_& TG;

  WALKER_TYPES walker_type = UNDEFINED_WALKER_TYPE;

  bool writer = false;
  bool accumulated_in_last_block = false;

  MixedObsHandler observ0;

  Wavefunction& wfn0;

  // Blocking info 
  int block_size   = 1;
  int iblock       = 0;
  int nblocks_skip = 0;

  bool write_metadata = true;
};
} // namespace afqmc
} // namespace qmcplusplus

#endif
