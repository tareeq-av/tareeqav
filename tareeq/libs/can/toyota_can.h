#pragma once

#include <iostream>

#include <cmath>
#include <memory>
#include <algorithm>

#include "packer.h"
#include "common.h"
#include "can_message.h"

namespace tareeq {
  namespace can {

class ToyotaCAN
{
private:
    double accel_ = 0.01;
    double previous_accel_ = 0.0;
    
    std::unique_ptr<CANPacker> packer_;


    class Limits {
        public:
            const double_t STEER_MAX        = 1500;
            const double_t STEER_DELTA_UP   = 10;      // 1.5s time to peak torque
            const double_t STEER_DELTA_DOWN = 25;      // always lower than 45 otherwise the Rav4 faults (Prius seems ok with 50)
            const double_t STEER_ERROR_MAX  = 350;     // max delta between torque cmd and torque motor
            // Accel limits
            const double ACCEL_HYST_GAP = 0.02;  // don't change accel command for small oscilalitons within this value
            const double ACCEL_MAX = 1.5;  // 1.5 m/s2
            const double ACCEL_MIN = -3.0; // 3   m/s2
            const double ACCEL_SCALE = std::max(ACCEL_MAX, -ACCEL_MIN);
    };

    const Limits LIMITS;

public:

    ToyotaCAN(std::unique_ptr<CANPacker>&& packer) : packer_(std::move(packer)){};

    double new_steering_command_ = 0;
    double prev_steering_command_ = 0;
    
    can_message create_accel_command(
        double accel_value, 
        double pcm_cancel,
        double standstill_req,
        double lead)
    {
        std::string name("ACC_CONTROL");

        std::map<std::string, double> values = {
            {"ACCEL_CMD", accel_value},
            {"SET_ME_X01", 1},
            {"DISTANCE", 0},
            {"MINI_CAR", lead},
            {"SET_ME_X3", 3},
            {"SET_ME_1", 1},
            {"RELEASE_STANDSTILL", not standstill_req},
            {"CANCEL_REQ", pcm_cancel}
        };

        return packer_->make_can_msg(name, values, -1);
    };

    can_message create_steering_command(
        double steer_torque_value, 
        double steer_request_on_off,
        uint32_t counter)
    {
        std::string name("STEERING_LKA");

        std::map<std::string, double> values = {
            {"LKA_STATE", 5.0},
            {"STEER_REQUEST" , 1.0},
            {"STEER_TORQUE_CMD" , steer_torque_value},
            {"SET_ME_1" , 1.0},
            {"COUNTER" , counter},
        };

        return packer_->make_can_msg(name, values, (int) counter);
    };

    uint8_t crc8_pedal(can_message message)
    {
        uint8_t crc = 0xFF;    // standard init value
        uint8_t poly = 0xD5;   // standard crc8: x8+x7+x6+x4+x2+1
        for (int i=(message.size-2); i >= 0; i--)
        {
            
            crc ^= message.data[i];
            // std::cout << "i == " << i << " and crc = " << +crc << std::endl;
            for (int j=0; j < 8; j++)
            {
                if ((crc & 0x80) != 0) {crc = ((crc << 1) ^ poly) & 0xFF;}
                else  {crc <<= 1;}
            }
        }
        // std::cout << std::endl;
        return crc;
    }

    can_message create_gas_command(double gas_amount, uint32_t counter)
    {
        std::string name("GAS_COMMAND");

        // common gas pedal msg generator
        uint8_t enable = gas_amount > 0.001;
        
        std::map<std::string, double> values = {
            {"ENABLE" , enable},
            {"COUNTER_PEDAL" , counter & 0xF}
        };

        if (enable)
        {
            values["GAS_COMMAND"] = gas_amount * 255.0;
            values["GAS_COMMAND2"] = gas_amount * 255.0;
        }

        can_message tmp = packer_->make_can_msg(name, values, -1);

        values["CHECKSUM_PEDAL"] = crc8_pedal(tmp);

        return packer_->make_can_msg(name, values, -1);
    }

    int32_t apply_steering_torque_limits(
       double apply_torque,
       double apply_torque_last,
       double motor_torque)
    {
        double max_lim = std::min(std::max(motor_torque + LIMITS.STEER_ERROR_MAX, LIMITS.STEER_ERROR_MAX), LIMITS.STEER_MAX);
        double min_lim = std::max(std::min(motor_torque - LIMITS.STEER_ERROR_MAX, -LIMITS.STEER_ERROR_MAX), -LIMITS.STEER_MAX);

        apply_torque = clip(apply_torque, min_lim, max_lim);

        if (apply_torque_last > 0)
        {
            apply_torque = clip(apply_torque,
                        std::max(apply_torque_last - LIMITS.STEER_DELTA_DOWN, -LIMITS.STEER_DELTA_UP),
                        apply_torque_last + LIMITS.STEER_DELTA_UP);
        }
        else
        {
            apply_torque = clip(apply_torque,
                        apply_torque_last - LIMITS.STEER_DELTA_UP,
                        std::min(apply_torque_last + LIMITS.STEER_DELTA_DOWN, LIMITS.STEER_DELTA_UP));
        }

        return int(std::round(float(apply_torque)));
    };

    inline void accel_hysteresis(double &accel, double &accel_steady)
    {
        if (accel > accel_steady + LIMITS.ACCEL_HYST_GAP)
            { accel = accel - LIMITS.ACCEL_HYST_GAP;}
        else if (accel < accel_steady - LIMITS.ACCEL_HYST_GAP)
            {accel = accel + LIMITS.ACCEL_HYST_GAP;}
    }

    can_message apply_accel_command(uint32_t counter)
    {
        double accel_steady = 0.0;
        accel_hysteresis(accel_, accel_steady);
        accel_ = clip(accel_ * LIMITS.ACCEL_SCALE, LIMITS.ACCEL_MIN, LIMITS.ACCEL_MAX);

        return create_accel_command(
            accel_,
            1, // TO-DO: create a variable for this
            0, // TO-DO: create a variable for this
            1 // TO-DO: create a variable for this
        );
    };

    can_message apply_steering_command(uint32_t counter)
    {
        // int32_t new_steering_command = int(std::round(steering_command_+1.0) * LIMITS.STEER_MAX);
        // new_steering_command_ = apply_steering_torque_limits(
        //     new_steering_command_,
        //     prev_steering_command_,
        //     500 // TO-DO: create a variable for this
        // );
        // if (new_steering_command_ == LIMITS.STEER_MAX)
        // {
        //     new_steering_command_ = 0;
        // }

        prev_steering_command_ = new_steering_command_;
        return create_steering_command(new_steering_command_, 1, counter);
    }
    
    can_message create_pcm_cruise_msg()
    {
        std::string name("PCM_CRUISE");

        std::map<std::string, double> values = {
            {"GAS_RELEASED", 0.0},
            {"CRUISE_ACTIVE" , 1.0},
            {"STANDSTILL_ON" , 0},
            {"ACCEL_NET" , 0},
            {"CRUISE_STATE" , 8.0},
            {"CANCEL_REQ" , 0},
        };

        return packer_->make_can_msg(name, values, -1);
    };

    can_message create_pcm_cruise_2_msg()
    {
        std::string name("PCM_CRUISE_2");

        std::map<std::string, double> values = {
            {"MAIN_ON", 0.0},
            {"LOW_SPEED_LOCKOUT" , 2.0},
            {"SET_SPEED" , 55.}
        };

        return packer_->make_can_msg(name, values, -1);
    };

    can_message create_wheel_speeds_msg()
    {
        std::string name("WHEEL_SPEEDS");

        std::map<std::string, double> values = {
            {"WHEEL_SPEED_FR", 47},
            {"WHEEL_SPEED_FL" , 47},
            {"WHEEL_SPEED_RR" , 47},
            {"WHEEL_SPEED_RL" , 47}
        };

        return packer_->make_can_msg(name, values, -1);
    };

    can_message create_esp_control_msg()
    {
        std::string name("ESP_CONTROL");

        std::map<std::string, double> values = {
            {"TC_DISABLED", 0.0},
            {"VSC_DISABLED" , 0.0},
            {"BRAKE_LIGHTS_ACC" , 0.0}
        };

        return packer_->make_can_msg(name, values, -1);
    };

    can_message create_seats_doors_msg()
    {
        std::string name("SEATS_DOORS");

        std::map<std::string, double> values = {
            {"SEATBELT_DRIVER_UNLATCHED", 0.0},
            {"DOOR_OPEN_FL" , 0.0},
            {"DOOR_OPEN_RL" , 0.0},
            {"DOOR_OPEN_RR" , 0.0},
            {"DOOR_OPEN_FR" , 0.0}
        };

        return packer_->make_can_msg(name, values, -1);
    };

    can_message create_gear_packet_msg()
    {
        std::string name("GEAR_PACKET");

        std::map<std::string, double> values = {
            {"GEAR", 0.0},
            {"SPORT_ON" , 0.0}
        };

        return packer_->make_can_msg(name, values, -1);
    };

    can_message create_gas_pedal_msg()
    {
        std::string name("GAS_PEDAL");

        std::map<std::string, double> values = {
            {"GAS_RELEASED", 0.0},
            {"GAS_PEDAL" , 0.0}
        };

        return packer_->make_can_msg(name, values, -1);
    };

    can_message create_brake_module_msg()
    {
        std::string name("BRAKE_MODULE");

        std::map<std::string, double> values = {
            {"BRAKE_PRESSURE", 0.0},
            {"BRAKE_PRESSED" , 0.0}
        };

        return packer_->make_can_msg(name, values, -1);
    };

    can_message create_eps_status_msg()
    {
        std::string name("EPS_STATUS");

        std::map<std::string, double> values = {
            {"IPAS_STATE", 3.0},
            {"LKA_STATE" , 5.0},
            {"TYPE", 1.0}
        };

        return packer_->make_can_msg(name, values, -1);
    };

    can_message steer_torque_sensor_msg()
    {
        std::string name("STEER_TORQUE_SENSOR");

        std::map<std::string, double> values = {
            {"STEER_TORQUE_EPS", 1000},
            {"STEER_TORQUE_DRIVER" , 1000},
            {"STEER_OVERRIDE", 1.0},
            {"STEER_ANGLE", 0.5}

        };

        return packer_->make_can_msg(name, values, -1);
    };
};

  } // namespace can
} // namespace tareeq