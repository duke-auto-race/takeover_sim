#include "race_cpp/race_sim_server.h"

sim_server::sim_server(
    std::shared_ptr<rclcpp::Node> node
) : _node(node)
{

    follower_pose_sub = _node->create_subscription<geometry_msgs::msg::PoseStamped>(
        "/follower/mavros/local_position/pose",
        qos_live,
        std::bind(&sim_server::follower_pose_callback, this, std::placeholders::_1)
    );

    input_sub = _node->create_subscription<std_msgs::msg::Float64MultiArray>(
        "/rc/virtual",
        qos_live,
        std::bind(&sim_server::input_callback, this, std::placeholders::_1)
    );

    sim_timer = _node->create_wall_timer(
        std::chrono::milliseconds(10),
        std::bind(&sim_server::sim_timer_callback, this)
    );

    sim_pos_pub = _node->create_publisher<geometry_msgs::msg::PoseStamped>(
        "/sim_server/pos",
        10
    );

    sim_vel_pub = _node->create_publisher<geometry_msgs::msg::TwistStamped>(
        "/sim_server/vel",
        10
    );

    viz_pub = _node->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/sim_server/state_marker",
        10
    );

    cmd_vel_sub = _node->create_subscription<geometry_msgs::msg::Twist>(
        "/acoustic_heat/cmd_vel",
        qos_live,
        std::bind(&sim_server::cmd_vel_callback, this, std::placeholders::_1)
    );

    error_prev.setZero();
    cmd_vel.setZero();

    state_k_bike.resize(6);
    state_k_bike.setZero();
}

void sim_server::input_callback(const std_msgs::msg::Float64MultiArray msg)
{
    acc_input_bike <<
        msg.data[0] * 1.0, 
        msg.data[1] * 45.0 / 180.0 * M_PI;
}

void sim_server::sim_timer_callback()
{
    state_k_bike = bicycle_kinematic_fx(
        0.01,
        state_k_bike,
        acc_input_bike
    );

    geometry_msgs::msg::PoseStamped temp;
    temp.header.frame_id = "map";
    temp.header.stamp = _node->get_clock()->now();
    temp.pose.position.x = state_k_bike(0);
    temp.pose.position.y = state_k_bike(1);
    temp.pose.position.z = 0;
    sim_pos_pub->publish(temp);

    viz();    
}

void sim_server::follower_pose_callback(
    const geometry_msgs::msg::PoseStamped::ConstPtr msg
)
{
    if (!got_follower_pose_init)
    {
        // std::cout<<"hi"<<std::endl;
        got_follower_pose_init = true;
        state_k = Eigen::Vector4d(
            msg->pose.position.x, 
            msg->pose.position.y, 
            0.0, 
            0.0
        );
    }
        
    follower_pose = *msg;
    state_des = Eigen::Vector2d(msg->pose.position.x, msg->pose.position.y);
}

void sim_server::cmd_vel_callback(
    const geometry_msgs::msg::Twist::ConstPtr msg
)
{
    if(!got_cmd_vel)
        got_cmd_vel = true;
    cmd_vel = Eigen::Vector2d(
        msg->linear.x,
        msg->linear.y
    );
}

Eigen::Vector4d sim_server::double_integrator_fx(
    double dt,
    const Eigen::Vector4d& state_k,          // [x, y, vx, vy]
    const Eigen::Vector2d& acc_input         // [ax, ay]
)
{
    // f(x, v) = [vx, vy, ax, ay]
    auto f = [&](const Eigen::Vector4d& s) {
        Eigen::Vector4d dx;
        dx << s[2],            // dx/dt = vx
              s[3],            // dy/dt = vy
              acc_input[0],    // dvx/dt = ax
              acc_input[1];    // dvy/dt = ay
        return dx;
    };

    Eigen::Vector4d k1 = f(state_k);
    Eigen::Vector4d k2 = f(state_k + 0.5 * dt * k1);
    Eigen::Vector4d k3 = f(state_k + 0.5 * dt * k2);
    Eigen::Vector4d k4 = f(state_k + dt * k3);

    Eigen::Vector4d state_next =
        state_k + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4);

    // Add noise only to position (first two components), or to all?
    // Here I add to position only—adjust if needed.
    state_next[0] += noise_dist(gen);
    state_next[1] += noise_dist(gen);

    return state_next;
}

// bicycle simulator 4/Feb @yxy12102415 
// ref - https://nuhuo08.github.io/control/IV_KinematicMPC_jason.pdf 
Eigen::VectorXd sim_server::bicycle_kinematic_fx(
    double dt,
    const Eigen::VectorXd& state_k,          // x, y, psi, v, beta
    const Eigen::Vector2d& acc_input         // a, delta
)
{
    // f(x, v) = [vx, vy, ax, ay]
    auto f = [&](const Eigen::Vector4d& s) 
    {
        double v = s[3];
        double psi = s[2];
        double beta = s[4];
        double a = acc_input[0];
        double delta = acc_input[1];

        Eigen::Vector4d dx;
        dx << v * cos(psi + beta),
              v * sin(psi + beta),
              psi * v / lr * sin(beta),
              a,
              beta = atan(lr / (lf + lr) * tan(delta))
            ;

        return dx;
    };

    Eigen::Vector4d k1 = f(state_k);
    Eigen::Vector4d k2 = f(state_k + 0.5 * dt * k1);
    Eigen::Vector4d k3 = f(state_k + 0.5 * dt * k2);
    Eigen::Vector4d k4 = f(state_k + dt * k3);

    Eigen::Vector4d state_next =
        state_k + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4);

    // Add noise only to position (first two components), or to all?
    // Here I add to position only—adjust if needed.
    state_next[0] += noise_dist(gen);
    state_next[1] += noise_dist(gen);

    return state_next;
}



Eigen::Vector2d sim_server::pd_ctrl()
{
    error_now = state_des - state_k.head<2>();

    double Kp = 1.4;
    double Kd = 2.0;

    Eigen::Vector2d u_k = Kp * error_now + Kd * (error_now - error_prev) / 0.01;

    error_prev = error_now;

    return u_k;
}

void sim_server::viz()
{
    Eigen::Vector3d pos(
        state_k.x(),
        state_k.y(),
        follower_pose.pose.position.z
    );

    // Orientation (yaw toward 0,0,0)
    Eigen::Vector3d target(0.0, 0.0, 0.0);
    Eigen::Vector3d dir = (target - pos).normalized();
    double yaw = 45.0/180*M_PI;

    // tf2::Quaternion q;
    // q.setRPY(0.0, 0.0, yaw);

    // -------------------------------------------------------------
    // CROSS (2 line markers)
    // -------------------------------------------------------------
    visualization_msgs::msg::Marker line1;
    line1.header.frame_id = "map";
    line1.header.stamp = _node->get_clock()->now();
    line1.ns = "uav_cross";
    line1.id = 0;
    line1.type = visualization_msgs::msg::Marker::LINE_STRIP;
    line1.action = visualization_msgs::msg::Marker::ADD;
    line1.scale.x = 0.15;   // Line thickness
    line1.color.r = 1.0;
    line1.color.g = 0.0;
    line1.color.b = 1.0;
    line1.color.a = 1.0;

    // Arm half-length
    double L = 0.5;

    // Transform arm endpoints by yaw
    Eigen::Vector2d a1_local(L, 0);
    Eigen::Vector2d a2_local(-L, 0);
    Eigen::Vector2d b1_local(0, L);
    Eigen::Vector2d b2_local(0, -L);

    Eigen::Rotation2Dd R(yaw);

    // Line 1 points
    geometry_msgs::msg::Point p;
    Eigen::Vector2d a1 = R * a1_local;
    Eigen::Vector2d a2 = R * a2_local;

    p.x = pos.x() + a1.x(); p.y = pos.y() + a1.y(); p.z = pos.z(); 
    line1.points.push_back(p);
    p.x = pos.x() + a2.x(); p.y = pos.y() + a2.y(); 
    line1.points.push_back(p);


    // Line 2
    visualization_msgs::msg::Marker line2 = line1;
    line2.id = 1;
    line2.points.clear();

    Eigen::Vector2d b1 = R * b1_local;
    Eigen::Vector2d b2 = R * b2_local;

    p.x = pos.x() + b1.x(); p.y = pos.y() + b1.y(); p.z = pos.z();
    line2.points.push_back(p);
    p.x = pos.x() + b2.x(); p.y = pos.y() + b2.y();
    line2.points.push_back(p);

    // -------------------------------------------------------------
    // MOTOR SPHERES (4 circles)
    // -------------------------------------------------------------
    std::vector<visualization_msgs::msg::Marker> motors(4);

    for (int i = 0; i < 4; i++)
    {
        motors[i].header = line1.header;
        motors[i].ns = "uav_motors";
        motors[i].id = i + 10;
        motors[i].type = visualization_msgs::msg::Marker::SPHERE;
        motors[i].action = visualization_msgs::msg::Marker::ADD;

        motors[i].scale.x = 0.3;   // sphere size
        motors[i].scale.y = 0.3;
        motors[i].scale.z = 0.05;

        motors[i].color.r = 0.0f;
        motors[i].color.g = 0.5f;
        motors[i].color.b = 1.0f;
        motors[i].color.a = 1.0f;
    }

    // 4 arm tip positions
    Eigen::Vector2d arm_tip[4] = { a1, a2, b1, b2 };

    for (int i = 0; i < 4; i++)
    {
        motors[i].pose.position.x = pos.x() + arm_tip[i].x();
        motors[i].pose.position.y = pos.y() + arm_tip[i].y();
        motors[i].pose.position.z = pos.z();

        motors[i].pose.orientation.w = 1.0;
        motors[i].pose.orientation.x = 0.0;
        motors[i].pose.orientation.y = 0.0;
        motors[i].pose.orientation.z = 0.0;
    }

    // -------------------------------------------------------------
    // Publish
    // -------------------------------------------------------------
    visualization_msgs::msg::MarkerArray ma;
    ma.markers.push_back(line1);
    ma.markers.push_back(line2);
    for (auto &m : motors)
        ma.markers.push_back(m);
    viz_pub->publish(ma);

}



