#include <iostream>
#include <chrono>
#include <thread>
#include <fstream>
#include <string>

#include "packer.h"
#include "parser.h"
#include "can_reader.h"
#include "toyota_can.h"
#include "can_message.h"
#include "toyota_corolla_2017.h"

void inline write_msg(
    std::ofstream &can_log_f_,
    tareeq::can::can_message &msg,
    std::unique_ptr<tareeq::can::CANParser> &parser
    )
{
    std::map<std::string, double> values = parser->parse(msg);
    can_log_f_ << msg.address << "," << msg.size << std::endl;
    for (const auto& kv : values)
    {
        can_log_f_ << kv.first << "," << kv.second << std::endl;
    }
    can_log_f_ << "--------------------" << "," << "--------------------" << std::endl;
}

int main(int argc, char** argv)
{
    std::vector<int> addresses;

    if (argc > 1)
    {
        for (int i=1; i < argc; i++)
        {
            addresses.push_back(std::atoi(argv[i]));
        }
    }

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
        
        if (addresses.empty())
        {
            write_msg(can_log_f_, msg, parser);
        }
        else if (std::find(addresses.begin(), addresses.end(), msg.address) != addresses.end())
        {
            write_msg(can_log_f_, msg, parser);
        }
    }

    return 0;

}

