#include <iostream>
#include <chrono>
#include <thread>
#include <fstream>

#include "packer.h"
#include "parser.h"
#include "can_reader.h"
#include "toyota_can.h"
#include "can_message.h"
#include "toyota_corolla_2017.h"


int main(int argc, char** argv)
{

    std::cout << "starting log file /home/pi/tareeqav/tareeq/libs/can/can_log.csv" << std::endl;
    std::ofstream can_log_f_;
    can_log_f_.open ("/home/pi/tareeqav/tareeq/libs/can/can_log.csv");

    dbc_register(&tareeq::can::toyota_corolla_2017_pt_generated);

    std::unique_ptr<tareeq::can::CANPacker> packer = tareeq::can::GetPacker(std::string("toyota_corolla_2017_pt_generated"));
    std::unique_ptr<tareeq::can::CANParser> parser = tareeq::can::GetParser(std::string("toyota_corolla_2017_pt_generated"));
    std::unique_ptr<tareeq::can::CANReader> reader = tareeq::can::GetReader(std::string("can1"));
    
    tareeq::can::ToyotaCAN toyota(std::move(packer));
    
    while (true)
    {
        tareeq::can::can_message msg = reader->receive();
        std::map<std::string, double> values = parser->parse(msg);

        can_log_f_ << msg.address << "," << msg.size << std::endl;
        for (const auto& kv : values)
        {
            can_log_f_ << kv.first << "," << kv.second << std::endl;
        }
        
        if (msg.address == 610)
        {
            // std::string binary = parser->get_binary_string(msg);
            // can_log_f_ << binary << "," << " " << std::endl;
            for (size_t i =0; i < msg.size; i++)
            {
                can_log_f_ << +msg.data[i];
            }
            can_log_f_ << "," << " " << std::endl;

        }
        can_log_f_ << "--------------------" << "," << "--------------------" << std::endl;
    }

    return 0;

}

