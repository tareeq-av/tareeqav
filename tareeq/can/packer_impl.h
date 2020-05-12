#pragma once

#include <cmath>
#include <memory>
#include <vector>
#include <cstring>
#include <cstdint>
#include <iostream>

#include "packer.h"
#include "can_base.h"
#include "can_message.h"

#include <bitset>


#define WARN printf

namespace tareeq {
  namespace can {

class CANPackerImpl : public CANPacker, public CANBase
{
  public:
    CANPackerImpl(const std::string& dbc_name) : CANBase(dbc_name){};

  private:
    
    /**
     * 
    */
    virtual uint64_t set_value(
        uint64_t ret,
        Signal sig,
        int64_t ival) 
    {
        int shift = sig.is_little_endian? sig.b1 : sig.bo;
        uint64_t mask = ((1ULL << sig.b2)-1) << shift;
        uint64_t dat = (ival & ((1ULL << sig.b2)-1)) << shift;

        // std::bitset<64> bmask(mask);
        // std::bitset<64> bdat(dat);
        // std::cout << "int value was " << ival << std::endl;
        // std::cout << "mask has binary " << bmask << std::endl;
        // std::cout << "dat  has binary " << bdat << std::endl;
        
        if (sig.is_little_endian) {
            dat = ReverseBytes(dat);
            mask = ReverseBytes(mask);
        }

        ret &= ~mask;

        // std::bitset<64> ret1(ret);
        // std::cout << "ret1 has binary" << ret1 << std::endl;

        ret |= dat;

        // std::bitset<64> ret2(ret);
        // std::cout << "ret2 has binary" << ret2 << std::endl;

        return ret;
    };

    /**
     * 
    */
    virtual uint64_t pack(
        uint32_t address,
        const std::vector<SignalPackValue> &signals,
        int counter)
    {
        
        uint64_t ret = 0;

        for (const auto& sigval : signals) {
            std::string name = std::string(sigval.name);
            double value = sigval.value;

            auto sig_it = signal_lookup.find(std::make_pair(address, name));
            if (sig_it == signal_lookup.end()) {
            WARN("undefined signal %s - %d\n", name.c_str(), address);
            continue;
            }
            auto sig = sig_it->second;

            int64_t ival = (int64_t)(round((value - sig.offset) / sig.factor));
            if (ival < 0) {
            ival = (1ULL << sig.b2) + ival;
            }

            std::cout << "processing " << name << std::endl;
            ret = set_value(ret, sig, ival);
            std::cout << std::endl << std::endl;
        }

        if (counter >= 0){
            auto sig_it = signal_lookup.find(std::make_pair(address, "COUNTER"));
            if (sig_it == signal_lookup.end()) {
            WARN("COUNTER not defined\n");
            return ret;
            }
            auto sig = sig_it->second;

            if ((sig.type != SignalType::HONDA_COUNTER) && (sig.type != SignalType::VOLKSWAGEN_COUNTER)) {
            WARN("COUNTER signal type not valid\n");
            }

            ret = set_value(ret, sig, counter);
        }

        auto sig_it_checksum = signal_lookup.find(std::make_pair(address, "CHECKSUM"));
        if (sig_it_checksum != signal_lookup.end()) {
            auto sig = sig_it_checksum->second;
            unsigned int chksm = toyota_checksum(address, ret, message_lookup[address].size);
            ret = set_value(ret, sig, chksm);
        }

        return ret;
    };

    virtual tareeq::can::can_message make_can_msg(
        std::string &name,
        std::map<std::string, double> &values,
        int counter)
    {
        /* message struct to be returned to caller */
        tareeq::can::can_message msg;
        std::vector<SignalPackValue> signals;

        for (const auto& kv : values)
        {
            SignalPackValue signal;

            std::string key = kv.first;
            double value = kv.second;
            
            signal.name = new char[key.length()+1];
            std::strcpy((char *)signal.name, key.c_str());
            signal.value = value;

            //add to list
            signals.push_back(signal);

        }

        std::pair<uint32_t, uint32_t> address_and_size = name_to_address_and_size[name];

        uint64_t val = pack(address_and_size.first, signals, -1);
        std::cout << "generated value " << val << " and as an array is:" << std::endl;
        val = ReverseBytes(val);

        std::cout << std::hex;
        std::cout << "reversed bytes value " << val << " and as an array is:" << std::endl;
        for (size_t i=0; i < address_and_size.second; i++) {
            std::cout << +(*(((uint8_t*)&val)+i)) << " " ;
        }

        std::cout << std::endl;
        std::memcpy(msg.data, &val, address_and_size.second);

        msg.address = address_and_size.first;
        msg.size = address_and_size.second;
        msg.bus = (uint32_t) 0;

        for (auto signal : signals)
        {
            delete signal.name;
            signal.name = NULL;
        }

        return msg;
    }
};

  } // namespace can
} // namespace tareeq

