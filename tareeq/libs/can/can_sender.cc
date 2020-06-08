#include <iostream>
#include <chrono>
#include <thread>

#include "packer.h"
#include "parser.h"
#include "can_writer.h"
#include "toyota_can.h"
#include "can_message.h"
#include "toyota_corolla_2017.h"


int main(int argc, char** argv)
{

    size_t frame_id = 0;
    dbc_register(&tareeq::can::toyota_corolla_2017_pt_generated);

    std::unique_ptr<tareeq::can::CANPacker> packer = tareeq::can::GetPacker(std::string("toyota_corolla_2017_pt_generated"));
    std::unique_ptr<tareeq::can::CANParser> parser = tareeq::can::GetParser(std::string("toyota_corolla_2017_pt_generated"));
    std::unique_ptr<tareeq::can::CANWriter> writer = tareeq::can::GetWriter(std::string("can0"));
    
    tareeq::can::ToyotaCAN toyota(std::move(packer));

    bool res;
    int steer_adjust = 0;
    toyota.new_steering_command_ = 200;

    // https://github.com/ros/ros_comm/blob/noetic-devel/clients/rospy/src/rospy/timer.py#L47
    // long sleep_duration = 2e4/100; // 1e9/100Hz 
    long sleep_duration = 1000000000.0/100;

    while (true)
    {
        auto s = std::chrono::system_clock::now();
        auto start_time = s.time_since_epoch();

        frame_id++;
        steer_adjust++;
        
        if (steer_adjust % 100 == 0)
        {
            if (toyota.new_steering_command_ >= 1200)
                {toyota.new_steering_command_ = 0;}
            else
            {toyota.new_steering_command_ += 75;}
        }

        frame_id++;
        tareeq::can::can_message steering_cmd = toyota.apply_steering_command(frame_id);
        
        frame_id++;
        tareeq::can::can_message accel_cmd    = toyota.apply_accel_command(frame_id);
        
        frame_id++;
        tareeq::can::can_message gas_cmd      = toyota.create_gas_command(0.5, frame_id);

        frame_id++;
        tareeq::can::can_message pcm_msg      = toyota.create_pcm_cruise_msg();

        frame_id++;
        tareeq::can::can_message pcm2_msg     = toyota.create_pcm_cruise_2_msg();

        frame_id++;
        tareeq::can::can_message whl_msg      = toyota.create_wheel_speeds_msg();

        frame_id++;
        tareeq::can::can_message esp_msg      = toyota.create_esp_control_msg();

        frame_id++;
        tareeq::can::can_message doors_msg    = toyota.create_seats_doors_msg();

        frame_id++;
        tareeq::can::can_message gear_msg     = toyota.create_gear_packet_msg();

        frame_id++;
        tareeq::can::can_message gas_msg      = toyota.create_gas_pedal_msg();

        frame_id++;
        tareeq::can::can_message brake_msg    = toyota.create_brake_module_msg();

        // frame_id++;
        // tareeq::can::can_message eps_msg      = toyota.create_eps_status_msg();

        // frame_id++;
        // tareeq::can::can_message steer_msg    = toyota.steer_torque_sensor_msg();

        res = writer->send(pcm_msg);
        res = writer->send(pcm2_msg);
        res = writer->send(whl_msg);
        res = writer->send(esp_msg);
        res = writer->send(doors_msg);
        res = writer->send(gear_msg);
        res = writer->send(gas_msg);
        res = writer->send(brake_msg);
        // res = writer->send(eps_msg);
        // res = writer->send(steer_msg);
        res = writer->send(steering_cmd);    
        res = writer->send(accel_cmd);
        res = writer->send(gas_cmd);

        if (!res)
        {
            std::cout << "unabe to send message" << std::endl;
        }

        if (frame_id % 10000 == 0)
        {
            std::cout << "we have sent " << frame_id << " messages" << std::endl; 
        }

        auto e = std::chrono::system_clock::now();
        auto end_time = e.time_since_epoch();

        auto exec_time = end_time.count() - start_time.count();
        if (exec_time > sleep_duration)
        {
            continue;
        }
        else 
        {
            auto sleep_for = sleep_duration - exec_time;
            std::cout << "execution took " << exec_time << " so sleeping for " << sleep_for << " nano seconds" << std::endl;
            std::this_thread::sleep_for(std::chrono::nanoseconds(sleep_for));
        }
        
        
    }

    return 0;

}

// int main(int argc, char** arg)
// {
//     std::chrono::time_point<std::chrono::system_clock> s = std::chrono::system_clock::now();
//     auto start = s.time_since_epoch();


//     std::this_thread::sleep_for(std::chrono::seconds(10));

//     std::chrono::time_point<std::chrono::system_clock> e = std::chrono::system_clock::now();
//     auto end = e.time_since_epoch();

//     std::cout << (end-start).count() << std::endl;
//     return 0;
// }