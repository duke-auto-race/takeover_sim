#include <chrono>
#include <memory>

#include "rclcpp/rclcpp.hpp"

#include <geometry_msgs/msg/pose_stamped.hpp>
#include "mavros_msgs/msg/state.hpp"
#include "mavros_msgs/srv/command_bool.hpp"
#include "mavros_msgs/srv/set_mode.hpp"

#include "race_cpp/race_sim_server.h"

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    std::shared_ptr<rclcpp::Node> node = rclcpp::Node::make_shared("SIM_SERVER");
    sim_server sim_server_node(node);

    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
