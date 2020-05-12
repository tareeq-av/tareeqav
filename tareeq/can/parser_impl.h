#pragma once

#include <cmath>
#include <memory>
#include <vector>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
#include <bitset>

#include "dbc.h"
#include "parser.h"
#include "can_base.h"
#include "can_message.h"


#define WARN printf

namespace tareeq {
  namespace can {

class CANParserImpl : public CANParser, public CANBase
{
  public:

    CANParserImpl(const std::string& dbc_name) : CANBase(dbc_name){};

    virtual std::map<std::string, double> parse(can_message can_msg)
    {
      std::map<std::string, double> values;

      std::string binary = get_binary_string(can_msg);
      Msg msg = message_lookup[can_msg.address];
      std::cout << "message has address " << msg.address << std::endl;
      for (size_t i=0; i<msg.num_sigs; i++) {
        Signal sig = msg.sigs[i];
        std::string value_str = binary.substr(sig.b1, sig.b2);
        values[sig.name] = std::stoi(value_str, nullptr, 2);
      }

      return values;
    }

  private:

    inline std::string int_to_hex(uint8_t i)
    {
        std::stringstream stream;
        stream << std::setfill ('0') << std::setw(sizeof(i)*2) 
                << std::hex << +i;
        return stream.str();
    }

    inline std::string get_binary_string(can_message can_msg)
    {
      std::string binary;
      for (size_t i=0; i < can_msg.size; i++)
      {
          std::string s = int_to_hex(can_msg.data[i]);
          std::stringstream ss;
          ss << std::hex << s;
          unsigned n;
          ss >> n;
          std::bitset<8> b(n);
          binary += b.to_string();
      }
      return binary;
    }
};

  } // namespace can
} // namespace tareeq

