#ifndef RACE_SIM_SERVER_LIB_H
#define RACE_SIM_SERVER_LIB_H

#include <chrono>
#include <memory>
#include <limits>
#include <cmath>

#include "rclcpp/rclcpp.hpp"

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/joy.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <eigen3/Eigen/Dense>
#include <visualization_msgs/msg/marker_array.hpp>
#include <tf2_ros/transform_broadcaster.h>
#define RAW_TRACK 0
#define TRAJ_TRACK 1

#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_msgs/msg/float64.hpp>

#include <random>
#include <mavros_msgs/msg/rc_in.hpp>

#include <std_msgs/msg/float32_multi_array.hpp>
// from std_msgs.msg import Float32MultiArray


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

        Eigen::VectorXd bicycle_dynamic_fx(
            double dt,
            const Eigen::VectorXd& state_k,          // [x, y, vx, vy]
            const Eigen::Vector2d& acc_input         // [ax, ay]
        );

        Eigen::VectorXd state_k_bike; // x, y, psi, v, beta
        Eigen::Vector2d acc_input_bike; // a, delta

        double lf = 1.6;  // front axle to CG (meters) - matches viz
        double lr = 1.6;  // rear axle to CG (meters) - matches viz
        double friction_coeff = 0.3; // friction coefficient for natural deceleration

        // UST-10LX LiDAR parameters
        double lidar_scan_angle = 270.0 * M_PI / 180.0;  // 270 degrees in radians
        double lidar_angular_resolution = 0.25 * M_PI / 180.0;  // 0.25 degrees in radians
        double lidar_max_range = 30.0;  // 30 meters
        double lidar_accuracy = 0.04;   // 40 mm
        double lidar_frequency = 40.0;  // 40 Hz
        int lidar_num_rays = int(lidar_scan_angle / lidar_angular_resolution); // ~1080 rays
        int lidar_scan_counter = 0;     // Counter for 40Hz publishing (every 2-3 iterations at 100Hz)

        // AI opponent vehicle
        double ai_path_position = 0.0;  // position along track [0, track_length]
        double ai_velocity = 10.0;  // constant velocity in m/s
        double track_total_length = 0.0;  // total track length
        void update_ai_vehicle(double dt);
        void get_track_pose(double s, double& x, double& y, double& psi);

        void input_callback(
            const std_msgs::msg::Float64MultiArray msg
        );

        rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr input_sub;

        rclcpp::TimerBase::SharedPtr sim_timer;    
        void sim_timer_callback();
        
        int track_publish_counter = 0;  // Counter for periodic track republishing

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
        // rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr sim_pos_pub;
        // rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr sim_vel_pub;
        rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr sim_pos_pub;
        rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr sim_vel_pub;
        rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr sim_vel_x_pub;
        rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr viz_pub;
        rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr track_pub;
        rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr ai_viz_pub;
        rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr lidar_viz_pub;
        rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr lidar_scan_pub;

        // tf broadcaster
        std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster;

        rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr rc_sub;

        void viz();
        void viz_track();
        void viz_ai_vehicle();
        void viz_lidar();
        void publish_lidar_scan(const rclcpp::Time& timestamp);
        
        // LiDAR ray intersection helpers
        double ray_line_intersection(double ray_x, double ray_y, double ray_dx, double ray_dy,
                                     double line_x1, double line_y1, double line_x2, double line_y2);
        double ray_circle_intersection(double ray_x, double ray_y, double ray_dx, double ray_dy,
                                       double circle_x, double circle_y, double circle_r);
        double ray_arc_intersection(double ray_x, double ray_y, double ray_dx, double ray_dy,
                                    double circle_x, double circle_y, double circle_r,
                                    double start_angle, double end_angle);
        double ray_box_intersection(double ray_x, double ray_y, double ray_dx, double ray_dy,
                                    double box_x, double box_y, double box_psi, 
                                    double box_width, double box_length);
        double compute_lidar_range(double ray_angle);
        
        geometry_msgs::msg::Quaternion mult_quat(
            const geometry_msgs::msg::Quaternion &a, 
            const geometry_msgs::msg::Quaternion &b);

        void rcCallback(const sensor_msgs::msg::Joy::ConstPtr msg);
        
    public:
        sim_server(std::shared_ptr<rclcpp::Node> node);
};

#endif