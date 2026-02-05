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

#include <std_msgs/msg/float64_multi_array.hpp>

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


        // bicycle simulator 4/Feb @yxy12102415 
        Eigen::VectorXd bicycle_kinematic_fx(
            double dt,
            const Eigen::VectorXd& state_k,          // [x, y, vx, vy]
            const Eigen::Vector2d& acc_input         // [ax, ay]
        );

        Eigen::VectorXd state_k_bike; // x, y, psi, v, beta
        Eigen::Vector2d acc_input_bike; // a, delta

        double lf = 0.16;
        double lr = 0.16;

        void input_callback(
            const std_msgs::msg::Float64MultiArray msg
        );

        rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr input_sub;

        rclcpp::TimerBase::SharedPtr sim_timer;    
        void sim_timer_callback();

        Eigen::Quaterniond rpy2q(const Eigen::Vector3d& rpy);
        static geometry_msgs::msg::Quaternion quat_from_rpy(double roll, double pitch, double yaw)
        {
            geometry_msgs::msg::Quaternion q;
            // convert half-angles
            double hr = roll  * 0.5;
            double hp = pitch * 0.5;
            double hy = yaw   * 0.5;

            double cr = std::cos(hr), sr = std::sin(hr);
            double cp = std::cos(hp), sp = std::sin(hp);
            double cy = std::cos(hy), sy = std::sin(hy);

            q.w = cr*cp*cy + sr*sp*sy;
            q.x = sr*cp*cy - cr*sp*sy;
            q.y = cr*sp*cy + sr*cp*sy;
            q.z = cr*cp*sy - sr*sp*cy;
            return q;
        }


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