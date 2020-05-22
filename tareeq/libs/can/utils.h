#pragma once

#include <map>
#include <iostream>
#include <algorithm>

#include "common.h"
#include "common_dbc.h"

namespace tareeq {
    namespace can {

/*
def clip(x, lo, hi):
    return max(lo, min(hi, x))
**/



    } // namespace can
} // namespace tareeq

/*
def create_steer_command(packer, steer, steer_req, raw_cnt):
  """Creates a CAN message for the Toyota Steer Command."""

  values = {
    "STEER_REQUEST": steer_req,
    "STEER_TORQUE_CMD": steer,
    "COUNTER": raw_cnt,
    "SET_ME_1": 1,
  }
  return packer.make_can_msg("STEERING_LKA", 2, values)
*/