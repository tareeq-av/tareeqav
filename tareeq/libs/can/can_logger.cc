// #include <iostream>

// // #include "packer.h"
// // #include "parser.h"
// // #include "toyota_can.h"
// // #include "can_message.h"
// // #include "toyota_corolla_2017.h"

// #include <iostream>
// #include <uavcan_linux/uavcan_linux.hpp>
// #include <uavcan/protocol/node_status_monitor.hpp>

// int backup()
// {

//     // dbc_register(&tareeq::can::toyota_corolla_2017_pt_generated);

//     // std::unique_ptr<tareeq::can::CANPacker> packer = tareeq::can::GetPacker(std::string("toyota_corolla_2017_pt_generated"));
//     // std::unique_ptr<tareeq::can::CANParser> parser = tareeq::can::GetParser(std::string("toyota_corolla_2017_pt_generated"));
    
//     // tareeq::can::ToyotaCAN toyota(std::move(packer));
//     // tareeq::can::can_message msg;

//     // for (size_t i=0; i < 1; i++)
//     // {
//     //     msg = toyota.create_steering_command(200, 1, 50);
//     // }
    
//     // std::cout << "returned message data" << std::endl;
//     // for (size_t i=0; i < msg.size; i++)
//     // {
//     //     std::cout << +msg.data[i] << " ";
//     // }
//     // std::cout << std::endl;

//     // std::cout << "\n\nnow attempting to decode a can_message struct objec\n\n" << std::endl;
//     // std::map<std::string, double> values = parser->parse(msg);

//     // for (const auto& kv : values)
//     // {
//     //     std::cout << "key " << kv.first << " has value " << kv.second << std::endl;
//     // }

//     return 0;

// }


// static uavcan_linux::NodePtr initNode(const std::vector<std::string>& ifaces, uavcan::NodeID nid,
//                                       const std::string& name)
// {
//     auto node = uavcan_linux::makeNode(ifaces);

//     /*
//      * Configuring the node.
//      */
//     node->setNodeID(nid);
//     node->setName(name.c_str());

//     node->getLogger().setLevel(uavcan::protocol::debug::LogLevel::DEBUG);

//     /*
//      * Starting the node.
//      */
//     std::cout << "Starting the node..." << std::endl;
//     const int start_res = node->start();
//     std::cout << "Start returned: " << start_res << std::endl;
//     // ENFORCE(0 == start_res);

//     std::cout << "Node started successfully" << std::endl;

//     /*
//      * Say Hi to the world.
//      */
//     node->setModeOperational();
//     node->logInfo("init", "Hello world! I'm [%*], NID %*",
//                   node->getNodeStatusProvider().getName().c_str(), int(node->getNodeID().get()));
//     return node;
// }


// int main(int argc, char** argv)
// {
//     return 0;
// }