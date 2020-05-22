#pragma once

#include <iostream>
#include <cstring>
#include <memory>

#include <unistd.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <sys/socket.h>

#include <linux/can.h>
#include <linux/can/raw.h>

namespace tareeq {
    namespace can {

class CANSocketBase {
    
protected:

    int32_t sock;
    struct sockaddr_can addr;
	struct ifreq ifr;
	struct can_frame frame;

    CANSocketBase(const std::string& ifname) {

        if ((sock = socket(PF_CAN, SOCK_RAW, CAN_RAW)) < 0) {
		    perror(std::string("Unable to open socket " + ifname).c_str());
	    }

        std::strcpy(ifr.ifr_name, ifname.c_str());
        ioctl(sock, SIOCGIFINDEX, &ifr);

        memset(&addr, 0, sizeof(addr));
        addr.can_family = AF_CAN;
        addr.can_ifindex = ifr.ifr_ifindex;

        if (bind(sock, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
            perror(std::string("Unable to bind to socket " + ifname).c_str());
        }
    };
};
    } // namespace can
} // namespace tareeq
