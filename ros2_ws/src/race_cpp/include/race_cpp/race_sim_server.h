#ifndef RACE_SIM_SERVER_LIB_H
#define RACE_SIM_SERVER_LIB_H

#include <chrono>
#include <memory>

#include "rclcpp/rclcpp.hpp"

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <eigen3/Eigen/Dense>
#include <visualization_msgs/msg/marker_array.hpp>
#define RAW_TRACK 0
#define TRAJ_TRACK 1

#include <random>

class sim_server
{
    private:
        std::shared_ptr<rclcpp::Node> _node;
        rclcpp::QoS qos_live = rclcpp::QoS(rclcpp::KeepLast(1))
            .best_effort()
            .durability_volatile();


        // sub: msg from px4
        rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr follower_pose_sub;
        void follower_pose_callback(const geometry_msgs::msg::PoseStamped::ConstPtr msg);
        geometry_msgs::msg::PoseStamped follower_pose;
        Eigen::Vector2d follower_pose_sim;
        bool got_follower_pose_init = false;

        // sub: msg from scout_relay
        rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub;
        void cmd_vel_callback(const geometry_msgs::msg::Twist::ConstPtr msg);
        Eigen::Vector2d cmd_vel;
        bool got_cmd_vel = false;

        // main sim
        Eigen::Vector4d double_integrator_fx(
            double dt,
            const Eigen::Vector4d& state_k,          // [x, y, vx, vy]
            const Eigen::Vector2d& acc_input         // [ax, ay]
        );
        Eigen::Vector4d state_k;

        rclcpp::TimerBase::SharedPtr sim_timer;    
        void sim_timer_callback();

        // sim noise
        std::default_random_engine gen;
        std::normal_distribution<double> noise_dist = std::normal_distribution<double>(0.0, 0.002);  // mean = 0, std = noise_std

        // ctrl pd
        Eigen::Vector2d error_now;
        Eigen::Vector2d error_prev;
        Eigen::Vector2d state_des;
        Eigen::Vector2d pd_ctrl();

        // pub
        rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr sim_pos_pub;
        rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr sim_vel_pub;
        rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr viz_pub;
        void viz();
        


        
    public:
        sim_server(std::shared_ptr<rclcpp::Node> node);
};

#endif