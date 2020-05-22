#pragma once

namespace tareeq {
    namespace can {

struct can_message
{
    uint32_t address;
    uint32_t bus;
    size_t size;
    uint8_t  data[8];
};

    } // namespace can
} // namespace tareeq
