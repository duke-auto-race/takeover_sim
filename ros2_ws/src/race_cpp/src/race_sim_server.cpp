#include "race_cpp/race_sim_server.h"

sim_server::sim_server(
    std::shared_ptr<rclcpp::Node> node
) : _node(node)
{
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

    rc_sub = _node->create_subscription<std_msgs::msg::Float32MultiArray>(
        "/rc/channels",
        10,
        std::bind(&sim_server::rcCallback, this, std::placeholders::_1)
    );

    state_k_bike.resize(5);
    state_k_bike.setZero();
}

void sim_server::input_callback(const std_msgs::msg::Float64MultiArray msg)
{
    acc_input_bike <<
        msg.data[0] * 1.0, 
        msg.data[1] * 20.0 / 180.0 * M_PI;
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

    Eigen::Quaterniond qtemp = rpy2q(
        Eigen::Vector3d(0, 0, state_k_bike(2))
    );

    std::cout<<state_k_bike<<std::endl<<std::endl;;

    temp.pose.orientation.w = qtemp.w();
    temp.pose.orientation.x = qtemp.x();
    temp.pose.orientation.y = qtemp.y();
    temp.pose.orientation.z = qtemp.z();


    sim_pos_pub->publish(temp);

    viz();    
}

Eigen::Quaterniond sim_server::rpy2q(const Eigen::Vector3d& rpy)
{
    Eigen::AngleAxisd rollAngle(rpy(0), Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitchAngle(rpy(1), Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yawAngle(rpy(2), Eigen::Vector3d::UnitZ());

    Eigen::Quaterniond q = yawAngle * pitchAngle * rollAngle;

    return q;
}


// bicycle simulator 4/Feb @yxy12102415 
// ref - https://nuhuo08.github.io/control/IV_KinematicMPC_jason.pdf 
Eigen::VectorXd sim_server::bicycle_kinematic_fx(
    double dt,
    const Eigen::VectorXd& state_k,          // x, y, psi, v, beta
    const Eigen::Vector2d& acc_input         // a, delta
)
{
    using namespace std;
    
    auto f = [&](const Eigen::VectorXd& s) 
    {
        using namespace std;
        double v = s[3];
        double psi = s[2];
        double beta = s[4];
        double a = acc_input[0];
        double delta = acc_input[1];


        Eigen::VectorXd dx;
        dx.resize(5);
        dx << v * cos(psi + beta),
              v * sin(psi + beta),
              v / lr * sin(beta),
              a,
              atan(lr / (lf + lr) * tan(delta))
        ;
        return dx;
    };

    Eigen::VectorXd k1 = f(state_k);
    Eigen::VectorXd k2 = f(state_k + 0.5 * dt * k1);
    Eigen::VectorXd k3 = f(state_k + 0.5 * dt * k2);
    Eigen::VectorXd k4 = f(state_k + dt * k3);

    Eigen::VectorXd state_next =
        state_k + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4);

    state_next[0] += noise_dist(gen);
    state_next[1] += noise_dist(gen);

    return state_next;
}

Eigen::VectorXd sim_server::bicycle_dynamic_fx(
    double dt,
    const Eigen::VectorXd& state_k,          // x, y, psi, v, beta
    const Eigen::Vector2d& acc_input         // a, delta
)
{
    using namespace std;
    
    auto f = [&](const Eigen::VectorXd& s) 
    {
        using namespace std;
        double v = s[3];
        double psi = s[2];
        double beta = s[4];
        double a = acc_input[0];
        double delta = acc_input[1];


        Eigen::VectorXd dx;
        dx.resize(5);
        dx << v * cos(psi + beta),
              v * sin(psi + beta),
              v / lr * sin(beta),
              a,
              atan(lr / (lf + lr) * tan(delta))
        ;
        return dx;
    };

    Eigen::VectorXd k1 = f(state_k);
    Eigen::VectorXd k2 = f(state_k + 0.5 * dt * k1);
    Eigen::VectorXd k3 = f(state_k + 0.5 * dt * k2);
    Eigen::VectorXd k4 = f(state_k + dt * k3);

    Eigen::VectorXd state_next =
        state_k + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4);

    state_next[0] += noise_dist(gen);
    state_next[1] += noise_dist(gen);

    return state_next;
}

void sim_server::viz()
{
    if (state_k_bike.size() != 5) {
        RCLCPP_WARN(_node->get_logger(), "viz: state_k_bike.size() != 5");
        return;
    }

    // geometry
    const double lf = 1.6;   // front axle to CG
    const double lr = 1.6;   // rear  axle to CG
    const double track = 1.6; // left-right distance
    const double wheel_radius = 0.6;
    const double wheel_thickness = 0.4; // cylinder height
    const double body_height = 1.0; // vehicle body vertical thickness
    const double body_clearance = wheel_radius; // body bottom sits at wheel top - wheel_radius -> adjust if needed

    // colors
    std_msgs::msg::ColorRGBA body_color; body_color.r=0.0; body_color.g=0.5; body_color.b=1.0; body_color.a=0.95;
    std_msgs::msg::ColorRGBA wheel_color; wheel_color.r=0.05; wheel_color.g=0.05; wheel_color.b=0.05; wheel_color.a=1.0;

    // state
    double x   = state_k_bike[0];
    double y   = state_k_bike[1];
    double psi = state_k_bike[2];
    // double v   = state_k_bike[3];
    // double beta= state_k_bike[4];
    double steer = acc_input_bike[1];

    // world z for vehicle (use follower_pose z as ground)
    double ground_z = follower_pose.pose.position.z;
    // center of body z: place body center above ground by wheel_radius + body_height/2
    double body_center_z = ground_z + wheel_radius + (body_height / 2.0);

    // headers
    std_msgs::msg::Header hdr;
    hdr.frame_id = "map";
    hdr.stamp = _node->get_clock()->now();

    // Body (CUBE)
    visualization_msgs::msg::Marker body_cube;
    body_cube.header = hdr;
    body_cube.ns = "car_body_3d";
    body_cube.id = 0;
    body_cube.type = visualization_msgs::msg::Marker::CUBE;
    body_cube.action = visualization_msgs::msg::Marker::ADD;

    // length = lf + lr; width = track; height = body_height
    body_cube.scale.x = lf + lr + 0.08; // add a bit to visually show bumper
    body_cube.scale.y = track + 0.06;
    body_cube.scale.z = body_height;
    body_cube.color = body_color;

    body_cube.pose.position.x = x;
    body_cube.pose.position.y = y;
    body_cube.pose.position.z = body_center_z;

    body_cube.pose.orientation = quat_from_rpy(0.0, 0.0, psi);

    visualization_msgs::msg::Marker wheels[4];
    for (int i=0;i<4;++i) {
        wheels[i].header = hdr;
        wheels[i].ns = "car_wheel_3d";
        wheels[i].id = 10 + i;
        wheels[i].type = visualization_msgs::msg::Marker::CYLINDER;
        wheels[i].action = visualization_msgs::msg::Marker::ADD;
        wheels[i].scale.x = wheel_thickness; // diameter along cylinder X (ignored for cylinder? using x/y to set diameter)
        wheels[i].scale.y = wheel_radius * 2.0;
        wheels[i].scale.z = wheel_radius * 2.0; // for some versions, scale.z is height; but for CYLINDER in visualization_msgs, scale.z is height (along local Z)
        // We'll set scale.x/scale.y to small and scale.z to wheel_thickness: but RViz interprets CYLINDER with scale.x, scale.y as diameters and scale.z as height.
        // Set as:
        wheels[i].scale.x = wheel_radius * 2.0; // diameter
        wheels[i].scale.y = wheel_radius * 2.0; // diameter
        wheels[i].scale.z = wheel_thickness;    // thickness of wheel
        wheels[i].color = wheel_color;
    }

    // wheel positions in vehicle frame (x, y) for centers
    // FL, FR, RL, RR
    Eigen::Vector2d w_fl(lf,  track/2.0);
    Eigen::Vector2d w_fr(lf, -track/2.0);
    Eigen::Vector2d w_rl(-lr,  track/2.0);
    Eigen::Vector2d w_rr(-lr, -track/2.0);
    std::vector<Eigen::Vector2d> wheel_positions = { w_fl, w_fr, w_rl, w_rr };

    // rotation for vehicle yaw
    Eigen::Rotation2Dd R(psi);

    for (int i=0;i<4;++i)
    {
        Eigen::Vector2d local = wheel_positions[i];
        Eigen::Vector2d world_xy = R * local; // rotate into map frame

        wheels[i].pose.position.x = x + world_xy.x();
        wheels[i].pose.position.y = y + world_xy.y();
        // wheel center z: ground_z + radius
        wheels[i].pose.position.z = ground_z + wheel_radius;

        // wheel orientation:
        // base: rotate cylinder by +90deg about X so cylinder axis aligns with vehicle Y
        // then apply yaw = psi (and steer for front)
        double wheel_yaw = psi;
        if (i == 0 || i == 1) wheel_yaw = psi + steer; // front wheels steered

        // Compose orientation: first rollX = +pi/2, then yaw = wheel_yaw
        geometry_msgs::msg::Quaternion q_rx90 = quat_from_rpy(M_PI/2.0, 0.0, 0.0);
        geometry_msgs::msg::Quaternion q_yaw = quat_from_rpy(0.0, 0.0, wheel_yaw);

        // quaternion multiplication q = q_yaw * q_rx90  (apply rx90 then yaw)
        geometry_msgs::msg::Quaternion q;
        // q = q_yaw * q_rx90
        // multiply (w1,x1,y1,z1) * (w2,x2,y2,z2)
        auto mult = [](const geometry_msgs::msg::Quaternion &a, const geometry_msgs::msg::Quaternion &b){
            geometry_msgs::msg::Quaternion r;
            r.w = a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z;
            r.x = a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y;
            r.y = a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x;
            r.z = a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w;
            return r;
        };
        q = mult(q_yaw, q_rx90);
        wheels[i].pose.orientation = q;
    }

    // Optionally draw an oriented arrow showing heading/velocity (along psi + beta if desired)
    visualization_msgs::msg::Marker heading_arrow;
    heading_arrow.header = hdr;
    heading_arrow.ns = "heading_3d";
    heading_arrow.id = 30;
    heading_arrow.type = visualization_msgs::msg::Marker::ARROW;
    heading_arrow.action = visualization_msgs::msg::Marker::ADD;
    heading_arrow.scale.x = 0.06; // shaft diameter
    heading_arrow.scale.y = 0.12; // head diameter
    heading_arrow.color.r = 1.0; heading_arrow.color.g = 0.2; heading_arrow.color.b = 0.2; heading_arrow.color.a = 0.9;

    geometry_msgs::msg::Point pa, pb;
    pa.x = x; pa.y = y; pa.z = body_center_z + 0.02;
    Eigen::Vector2d head = Eigen::Rotation2Dd(psi) * Eigen::Vector2d(0.40, 0.0);
    pb.x = x + head.x();
    pb.y = y + head.y();
    pb.z = pa.z;
    heading_arrow.points.push_back(pa);
    heading_arrow.points.push_back(pb);

    // Build MarkerArray and publish
    visualization_msgs::msg::MarkerArray ma;
    ma.markers.push_back(body_cube);
    for (int i=0;i<4;++i) ma.markers.push_back(wheels[i]);
    ma.markers.push_back(heading_arrow);

    viz_pub->publish(ma);
}

void sim_server::rcCallback(const std_msgs::msg::Float32MultiArray::ConstPtr msg)
{
    acc_input_bike <<
        msg->data[2] * 1.0, 
        msg->data[0] * 20.0 / 180.0 * M_PI;
    
}
