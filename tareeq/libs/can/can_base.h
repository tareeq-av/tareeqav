#pragma once

#include <map>
#include <memory>
#include <cassert>

#include "dbc.h"
#include "common.h"

namespace tareeq {
    namespace can {

class CANBase {
    
protected:

    const DBC* dbc = NULL;
    
    std::map<std::pair<uint32_t, std::string>, Signal> signal_lookup;
    std::map<uint32_t, Msg> message_lookup;
    std::map<std::string, std::pair<uint32_t, uint32_t>> name_to_address_and_size;
    std::map<uint32_t, uint32_t> address_to_size;

    uint64_t ReverseBytes(uint64_t x) {
        return ((x & 0xff00000000000000ull) >> 56) |
          ((x & 0x00ff000000000000ull) >> 40) |
          ((x & 0x0000ff0000000000ull) >> 24) |
          ((x & 0x000000ff00000000ull) >> 8) |
          ((x & 0x00000000ff000000ull) << 8) |
          ((x & 0x0000000000ff0000ull) << 24) |
          ((x & 0x000000000000ff00ull) << 40) |
          ((x & 0x00000000000000ffull) << 56);
    };

    CANBase(const std::string& dbc_name) {
        dbc = dbc_lookup(dbc_name);
        assert(dbc);
        for (size_t i=0; i<dbc->num_msgs; i++)
        {
            const Msg* msg = &dbc->msgs[i];
            message_lookup[msg->address] = *msg;
            for (size_t j=0; j<msg->num_sigs; j++) {
                const Signal* sig = &msg->sigs[j];
                signal_lookup[std::make_pair(msg->address, std::string(sig->name))] = *sig;
            }
        }
        init_crc_lookup_tables();

        size_t num_msgs = dbc[0].num_msgs;
        for (size_t i=0; i < num_msgs; i++)
        {
            Msg msg = dbc[0].msgs[i];
            name_to_address_and_size[msg.name] = std::make_pair(msg.address, msg.size);
            address_to_size[msg.address] = msg.size;
        }
    };
};

// inline CANBase::~CANBase() {};

    } // namespace can
} // namespace tareeq
