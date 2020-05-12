#include <iostream>

#include "packer.h"
#include "parser.h"
#include "toyota_can.h"
#include "can_message.h"
#include "toyota_corolla_2017.h"

int main(int argc, char** argv)
{

    dbc_register(&tareeq::can::toyota_corolla_2017_pt_generated);

    std::unique_ptr<tareeq::can::CANPacker> packer = tareeq::can::GetPacker(std::string("toyota_corolla_2017_pt_generated"));
    std::unique_ptr<tareeq::can::CANParser> parser = tareeq::can::GetParser(std::string("toyota_corolla_2017_pt_generated"));
    
    tareeq::can::ToyotaCAN toyota(std::move(packer));
    tareeq::can::can_message msg;

    for (size_t i=0; i < 1; i++)
    {
        msg = toyota.create_steering_command(200, 1, 50);
    }
    
    std::cout << "returned message data" << std::endl;
    for (size_t i=0; i < msg.size; i++)
    {
        std::cout << +msg.data[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "\n\nnow attempting to decode a can_message struct objec\n\n" << std::endl;
    std::map<std::string, double> values = parser->parse(msg);

    for (const auto& kv : values)
    {
        std::cout << "key " << kv.first << " has value " << kv.second << std::endl;
    }

    return 0;

}
