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

    sim_vel_x_pub = _node->create_publisher<std_msgs::msg::Float64>(
        "/sim_server/vel_x",
        10
    );

    viz_pub = _node->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/sim_server/state_marker",
        10
    );

    track_pub = _node->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/sim_server/track_marker",
        10
    );

    ai_viz_pub = _node->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/sim_server/ai_marker",
        10
    );

    ai_pos_pub = _node->create_publisher<geometry_msgs::msg::PoseStamped>(
        "/sim_server/ai_pos",
        10
    );

    lidar_viz_pub = _node->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/sim_server/lidar_marker",
        10
    );

    lidar_scan_pub = _node->create_publisher<sensor_msgs::msg::LaserScan>(
        "/sim_server/scan",
        10
    );

    tf_broadcaster = std::make_shared<tf2_ros::TransformBroadcaster>(_node);

    rc_sub = _node->create_subscription<sensor_msgs::msg::Joy>(
        "/joy",
        10,
        std::bind(&sim_server::rcCallback, this, std::placeholders::_1)
    );

    state_k_bike.resize(5);
    state_k_bike.setZero();

    // Calculate total track length for AI vehicle
    const double track_length = 100.0;
    const double turn_radius = 40.0;
    track_total_length = 2.0 * track_length + 2.0 * M_PI * turn_radius;
    ai_path_position = 0.0;  // Start at beginning

    // Initialize player vehicle at track starting position
    // state_k_bike: [x, y, psi, v, beta]
    state_k_bike[0] = -track_length/2.0;  // x: start of bottom straight
    state_k_bike[1] = -turn_radius;        // y: bottom straight position
    state_k_bike[2] = 0.0;                 // psi: heading in +x direction
    state_k_bike[3] = 0.0;                 // v: velocity
    state_k_bike[4] = 0.0;                 // beta: slip angle

    _node->declare_parameter("ai_vel", 0.0);
    ai_velocity = _node->get_parameter("ai_vel").as_double();

    _node->declare_parameter("auto", 0.0);
    ai_velocity = _node->get_parameter("ai_vel").as_double();
    std::cout<<ai_velocity<<std::endl;

    // Publish track once at startup
    viz_track();
}

void sim_server::input_callback(const std_msgs::msg::Float64MultiArray msg)
{
    acc_input_bike <<
        msg.data[0] * 1.0, 
        msg.data[1] * 20.0 / 180.0 * M_PI;

    
}

void sim_server::sim_timer_callback()
{
    // Get current timestamp once for consistent timing across all messages and TFs
    auto current_time = _node->get_clock()->now();
    
    state_k_bike = bicycle_kinematic_fx(
        0.01,
        state_k_bike,
        acc_input_bike
    );

    geometry_msgs::msg::PoseStamped temp;
    temp.header.frame_id = "map";
    temp.header.stamp = current_time;
    temp.pose.position.x = state_k_bike(0);
    temp.pose.position.y = state_k_bike(1);
    temp.pose.position.z = 0;

    Eigen::Quaterniond qtemp = rpy2q(
        Eigen::Vector3d(0, 0, state_k_bike(2))
    );

    // std::cout<<state_k_bike<<std::endl<<std::endl;;

    temp.pose.orientation.w = qtemp.w();
    temp.pose.orientation.x = qtemp.x();
    temp.pose.orientation.y = qtemp.y();
    temp.pose.orientation.z = qtemp.z();


    sim_pos_pub->publish(temp);

    // Publish velocity
    geometry_msgs::msg::TwistStamped vel_msg;
    vel_msg.header.frame_id = "base_link";
    vel_msg.header.stamp = current_time;
    // state_k_bike: [x, y, psi, v, beta]
    // Velocity in body frame: v is forward speed, beta is slip angle
    double v = state_k_bike(3);
    double beta = state_k_bike(4);
    // Transform velocity to body frame considering slip angle
    vel_msg.twist.linear.x = v * std::cos(beta);  // Forward velocity
    vel_msg.twist.linear.y = v * std::sin(beta);  // Lateral velocity (slip)
    vel_msg.twist.linear.z = 0.0;
    // Angular velocity is rate of change of heading (approximated from steering)
    vel_msg.twist.angular.x = 0.0;
    vel_msg.twist.angular.y = 0.0;
    vel_msg.twist.angular.z = v * std::tan(acc_input_bike(1)) / (lf + lr);  // Yaw rate
    sim_vel_pub->publish(vel_msg);

    // Publish x-axis velocity (forward velocity in base_link frame)
    std_msgs::msg::Float64 vel_x_msg;
    vel_x_msg.data = v * std::cos(beta);  // Forward velocity component
    sim_vel_x_pub->publish(vel_x_msg);

    // Broadcast TF transform
    geometry_msgs::msg::TransformStamped transform;
    transform.header.stamp = current_time;
    transform.header.frame_id = "map";
    transform.child_frame_id = "base_link";
    transform.transform.translation.x = state_k_bike(0);
    transform.transform.translation.y = state_k_bike(1);
    transform.transform.translation.z = 0.0;
    transform.transform.rotation.w = qtemp.w();
    transform.transform.rotation.x = qtemp.x();
    transform.transform.rotation.y = qtemp.y();
    transform.transform.rotation.z = qtemp.z();
    tf_broadcaster->sendTransform(transform);

    // Broadcast camera frame for third-person view
    geometry_msgs::msg::TransformStamped camera_transform;
    camera_transform.header.stamp = current_time;
    camera_transform.header.frame_id = "base_link";
    camera_transform.child_frame_id = "camera_link";
    // Position camera behind and above the vehicle
    camera_transform.transform.translation.x = -5.0;  // 5m behind
    camera_transform.transform.translation.y = 0.0;
    camera_transform.transform.translation.z = 3.0;   // 3m above
    // Rotate camera to look forward at the vehicle
    Eigen::Quaterniond cam_quat = rpy2q(Eigen::Vector3d(0.0, 0.5, 0.0));  // pitch down slightly
    camera_transform.transform.rotation.w = cam_quat.w();
    camera_transform.transform.rotation.x = cam_quat.x();
    camera_transform.transform.rotation.y = cam_quat.y();
    camera_transform.transform.rotation.z = cam_quat.z();
    tf_broadcaster->sendTransform(camera_transform);

    // Republish track every 1 second (100 iterations at 10ms)
    track_publish_counter++;
    if (track_publish_counter >= 100) {
        viz_track();
        track_publish_counter = 0;
    }

    // Update and visualize AI opponent vehicle
    update_ai_vehicle(0.01);
    viz_ai_vehicle();

    // Publish LiDAR scan at 40Hz (timer is 100Hz, so publish every 2-3 iterations)
    lidar_scan_counter++;
    if (lidar_scan_counter >= 3) {  // Approximately 40Hz (100/2.5 = 40)
        publish_lidar_scan(current_time);
        lidar_scan_counter = 0;
    } else if (lidar_scan_counter == 2) {
        // Publish every other time at counter=2 to achieve closer to 40Hz
        static int skip_toggle = 0;
        if (skip_toggle % 2 == 0) {
            publish_lidar_scan(current_time);
            lidar_scan_counter = 0;
        }
        skip_toggle++;
    }

    viz_lidar();
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
              a - friction_coeff * v,  // friction opposes velocity
              0.0  // beta is computed algebraically, not integrated
        ;
        return dx;
    };

    Eigen::VectorXd k1 = f(state_k);
    Eigen::VectorXd k2 = f(state_k + 0.5 * dt * k1);
    Eigen::VectorXd k3 = f(state_k + 0.5 * dt * k2);
    Eigen::VectorXd k4 = f(state_k + dt * k3);

    Eigen::VectorXd state_next =
        state_k + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4);

    // Compute beta directly from steering angle (algebraic constraint)
    state_next[4] = atan(lr / (lf + lr) * tan(acc_input[1]));

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
              a - friction_coeff * v,  // friction opposes velocity
              0.0  // beta is computed algebraically, not integrated
        ;
        return dx;
    };

    Eigen::VectorXd k1 = f(state_k);
    Eigen::VectorXd k2 = f(state_k + 0.5 * dt * k1);
    Eigen::VectorXd k3 = f(state_k + 0.5 * dt * k2);
    Eigen::VectorXd k4 = f(state_k + dt * k3);

    Eigen::VectorXd state_next =
        state_k + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4);

    // Compute beta directly from steering angle (algebraic constraint)
    state_next[4] = atan(lr / (lf + lr) * tan(acc_input[1]));

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
    double steer = acc_input_bike[1];

    // world z for vehicle (use follower_pose z as ground)
    double ground_z = 0.0;  // relative to base_link
    // center of body z: place body center above ground by wheel_radius + body_height/2
    double body_center_z = ground_z + wheel_radius + (body_height / 2.0);

    // headers - now in base_link frame
    std_msgs::msg::Header hdr;
    hdr.frame_id = "base_link";
    hdr.stamp = _node->get_clock()->now();

    // Body (CUBE) - at origin in base_link frame
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

    body_cube.pose.position.x = 0.0;
    body_cube.pose.position.y = 0.0;
    body_cube.pose.position.z = body_center_z;

    body_cube.pose.orientation = quat_from_rpy(0.0, 0.0, 0.0);  // no rotation in base_link frame

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

    for (int i=0;i<4;++i)
    {
        Eigen::Vector2d local = wheel_positions[i];

        wheels[i].pose.position.x = local.x();
        wheels[i].pose.position.y = local.y();
        // wheel center z: ground_z + radius
        wheels[i].pose.position.z = ground_z + wheel_radius;

        // wheel orientation:
        // base: rotate cylinder by +90deg about X so cylinder axis aligns with vehicle Y
        // then apply steer for front wheels (no psi needed, already in base_link)
        double wheel_yaw = 0.0;
        if (i == 0 || i == 1) wheel_yaw = steer; // front wheels steered

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
    pa.x = 0.0; pa.y = 0.0; pa.z = body_center_z + 0.02;
    pb.x = 0.40; pb.y = 0.0; pb.z = pa.z;  // arrow pointing forward in base_link frame
    heading_arrow.points.push_back(pa);
    heading_arrow.points.push_back(pb);

    // Build MarkerArray and publish
    visualization_msgs::msg::MarkerArray ma;
    ma.markers.push_back(body_cube);
    for (int i=0;i<4;++i) ma.markers.push_back(wheels[i]);
    ma.markers.push_back(heading_arrow);

    viz_pub->publish(ma);
}

void sim_server::viz_track()
{
    visualization_msgs::msg::MarkerArray track_markers;
    
    // Track parameters - NASCAR-style oval
    const double track_length = 100.0;  // straight section length
    const double track_width = 12.0;    // road width
    const double turn_radius = 40.0;    // radius of curved sections
    const double road_thickness = 0.05; // thickness of road surface
    
    std_msgs::msg::Header hdr;
    hdr.frame_id = "map";
    hdr.stamp = _node->get_clock()->now();
    
    // Grey road color
    std_msgs::msg::ColorRGBA road_color;
    road_color.r = 0.4; road_color.g = 0.4; road_color.b = 0.4; road_color.a = 1.0;
    
    // White line color for starting line
    std_msgs::msg::ColorRGBA white_color;
    white_color.r = 1.0; white_color.g = 1.0; white_color.b = 1.0; white_color.a = 1.0;
    
    // Red line color for starting line
    std_msgs::msg::ColorRGBA red_color;
    red_color.r = 1.0; red_color.g = 0.0; red_color.b = 0.0; red_color.a = 1.0;
    
    int marker_id = 0;
    
    // Bottom straight section
    visualization_msgs::msg::Marker bottom_straight;
    bottom_straight.header = hdr;
    bottom_straight.ns = "track";
    bottom_straight.id = marker_id++;
    bottom_straight.type = visualization_msgs::msg::Marker::CUBE;
    bottom_straight.action = visualization_msgs::msg::Marker::ADD;
    bottom_straight.pose.position.x = 0.0;
    bottom_straight.pose.position.y = -turn_radius;
    bottom_straight.pose.position.z = -road_thickness/2.0;
    bottom_straight.pose.orientation.w = 1.0;
    bottom_straight.scale.x = track_length;
    bottom_straight.scale.y = track_width;
    bottom_straight.scale.z = road_thickness;
    bottom_straight.color = road_color;
    track_markers.markers.push_back(bottom_straight);
    
    // Top straight section
    visualization_msgs::msg::Marker top_straight;
    top_straight.header = hdr;
    top_straight.ns = "track";
    top_straight.id = marker_id++;
    top_straight.type = visualization_msgs::msg::Marker::CUBE;
    top_straight.action = visualization_msgs::msg::Marker::ADD;
    top_straight.pose.position.x = 0.0;
    top_straight.pose.position.y = turn_radius;
    top_straight.pose.position.z = -road_thickness/2.0;
    top_straight.pose.orientation.w = 1.0;
    top_straight.scale.x = track_length;
    top_straight.scale.y = track_width;
    top_straight.scale.z = road_thickness;
    top_straight.color = road_color;
    track_markers.markers.push_back(top_straight);
    
    // Right curved section (using multiple small cubes to approximate curve)
    int num_segments = 30;
    for (int i = 0; i < num_segments; ++i) {
        double angle = -M_PI/2.0 + (M_PI * i / (num_segments - 1));
        double x = track_length/2.0 + turn_radius * cos(angle);
        double y = turn_radius * sin(angle);
        double yaw = angle + M_PI/2.0;
        
        visualization_msgs::msg::Marker curve_segment;
        curve_segment.header = hdr;
        curve_segment.ns = "track";
        curve_segment.id = marker_id++;
        curve_segment.type = visualization_msgs::msg::Marker::CUBE;
        curve_segment.action = visualization_msgs::msg::Marker::ADD;
        curve_segment.pose.position.x = x;
        curve_segment.pose.position.y = y;
        curve_segment.pose.position.z = -road_thickness/2.0;
        curve_segment.pose.orientation = quat_from_rpy(0.0, 0.0, yaw);
        curve_segment.scale.x = (M_PI * turn_radius / num_segments) * 1.2;
        curve_segment.scale.y = track_width;
        curve_segment.scale.z = road_thickness;
        curve_segment.color = road_color;
        track_markers.markers.push_back(curve_segment);
    }
    
    // Left curved section
    for (int i = 0; i < num_segments; ++i) {
        double angle = M_PI/2.0 + (M_PI * i / (num_segments - 1));
        double x = -track_length/2.0 + turn_radius * cos(angle);
        double y = turn_radius * sin(angle);
        double yaw = angle + M_PI/2.0;
        
        visualization_msgs::msg::Marker curve_segment;
        curve_segment.header = hdr;
        curve_segment.ns = "track";
        curve_segment.id = marker_id++;
        curve_segment.type = visualization_msgs::msg::Marker::CUBE;
        curve_segment.action = visualization_msgs::msg::Marker::ADD;
        curve_segment.pose.position.x = x;
        curve_segment.pose.position.y = y;
        curve_segment.pose.position.z = -road_thickness/2.0;
        curve_segment.pose.orientation = quat_from_rpy(0.0, 0.0, yaw);
        curve_segment.scale.x = (M_PI * turn_radius / num_segments) * 1.2;
        curve_segment.scale.y = track_width;
        curve_segment.scale.z = road_thickness;
        curve_segment.color = road_color;
        track_markers.markers.push_back(curve_segment);
    }
    
    // Starting line (checkered pattern - alternating red and white)
    int num_checks = 10;
    double check_width = track_width / num_checks;
    for (int i = 0; i < num_checks; ++i) {
        visualization_msgs::msg::Marker start_line;
        start_line.header = hdr;
        start_line.ns = "track";
        start_line.id = marker_id++;
        start_line.type = visualization_msgs::msg::Marker::CUBE;
        start_line.action = visualization_msgs::msg::Marker::ADD;
        start_line.pose.position.x = -track_length/2.0;
        start_line.pose.position.y = -turn_radius + (i * check_width) + check_width/2.0 - track_width/2.0;
        start_line.pose.position.z = 0.02;  // slightly above road
        start_line.pose.orientation.w = 1.0;
        start_line.scale.x = 1.0;  // 1m wide line
        start_line.scale.y = check_width;
        start_line.scale.z = road_thickness;
        start_line.color = (i % 2 == 0) ? white_color : red_color;
        track_markers.markers.push_back(start_line);
    }
    
    // Lane markings (center dashed line for entire track)
    double dash_length = 3.0;  // 3m dashes
    double dash_gap = 2.0;     // 2m gaps
    double dash_width = 0.3;   // width of dashes
    
    // Dashes on bottom straight
    int num_bottom_dashes = int(track_length / (dash_length + dash_gap));
    for (int i = 0; i < num_bottom_dashes; ++i) {
        visualization_msgs::msg::Marker dash;
        dash.header = hdr;
        dash.ns = "track";
        dash.id = marker_id++;
        dash.type = visualization_msgs::msg::Marker::CUBE;
        dash.action = visualization_msgs::msg::Marker::ADD;
        dash.pose.position.x = -track_length/2.0 + i * (dash_length + dash_gap) + dash_length/2.0;
        dash.pose.position.y = -turn_radius;
        dash.pose.position.z = 0.01;
        dash.pose.orientation.w = 1.0;
        dash.scale.x = dash_length;
        dash.scale.y = dash_width;
        dash.scale.z = road_thickness;
        dash.color = white_color;
        track_markers.markers.push_back(dash);
    }
    
    // Dashes on top straight
    int num_top_dashes = int(track_length / (dash_length + dash_gap));
    for (int i = 0; i < num_top_dashes; ++i) {
        visualization_msgs::msg::Marker dash;
        dash.header = hdr;
        dash.ns = "track";
        dash.id = marker_id++;
        dash.type = visualization_msgs::msg::Marker::CUBE;
        dash.action = visualization_msgs::msg::Marker::ADD;
        dash.pose.position.x = track_length/2.0 - i * (dash_length + dash_gap) - dash_length/2.0;
        dash.pose.position.y = turn_radius;
        dash.pose.position.z = 0.01;
        dash.pose.orientation.w = 1.0;
        dash.scale.x = dash_length;
        dash.scale.y = dash_width;
        dash.scale.z = road_thickness;
        dash.color = white_color;
        track_markers.markers.push_back(dash);
    }
    
    // Dashes on right curved section
    double right_curve_length = M_PI * turn_radius;
    int num_right_curve_dashes = int(right_curve_length / (dash_length + dash_gap));
    for (int i = 0; i < num_right_curve_dashes; ++i) {
        double arc_distance = i * (dash_length + dash_gap) + dash_length/2.0;
        double angle = -M_PI/2.0 + (arc_distance / turn_radius);
        double x = track_length/2.0 + turn_radius * cos(angle);
        double y = turn_radius * sin(angle);
        double yaw = angle + M_PI/2.0;
        
        visualization_msgs::msg::Marker dash;
        dash.header = hdr;
        dash.ns = "track";
        dash.id = marker_id++;
        dash.type = visualization_msgs::msg::Marker::CUBE;
        dash.action = visualization_msgs::msg::Marker::ADD;
        dash.pose.position.x = x;
        dash.pose.position.y = y;
        dash.pose.position.z = 0.01;
        dash.pose.orientation = quat_from_rpy(0.0, 0.0, yaw);
        dash.scale.x = dash_length;
        dash.scale.y = dash_width;
        dash.scale.z = road_thickness;
        dash.color = white_color;
        track_markers.markers.push_back(dash);
    }
    
    // Dashes on left curved section
    double left_curve_length = M_PI * turn_radius;
    int num_left_curve_dashes = int(left_curve_length / (dash_length + dash_gap));
    for (int i = 0; i < num_left_curve_dashes; ++i) {
        double arc_distance = i * (dash_length + dash_gap) + dash_length/2.0;
        double angle = M_PI/2.0 + (arc_distance / turn_radius);
        double x = -track_length/2.0 + turn_radius * cos(angle);
        double y = turn_radius * sin(angle);
        double yaw = angle + M_PI/2.0;
        
        visualization_msgs::msg::Marker dash;
        dash.header = hdr;
        dash.ns = "track";
        dash.id = marker_id++;
        dash.type = visualization_msgs::msg::Marker::CUBE;
        dash.action = visualization_msgs::msg::Marker::ADD;
        dash.pose.position.x = x;
        dash.pose.position.y = y;
        dash.pose.position.z = 0.01;
        dash.pose.orientation = quat_from_rpy(0.0, 0.0, yaw);
        dash.scale.x = dash_length;
        dash.scale.y = dash_width;
        dash.scale.z = road_thickness;
        dash.color = white_color;
        track_markers.markers.push_back(dash);
    }
    
    track_pub->publish(track_markers);
    // RCLCPP_INFO(_node->get_logger(), "Track visualization published with %zu markers", track_markers.markers.size());
}

void sim_server::update_ai_vehicle(double dt)
{
    // Move AI vehicle along track at constant velocity
    ai_path_position += ai_velocity * dt;
    
    // Wrap around track
    if (ai_path_position >= track_total_length) {
        ai_path_position -= track_total_length;
    }
}

void sim_server::get_track_pose(double s, double& x, double& y, double& psi)
{
    // Track parameters - must match viz_track()
    const double track_length = 100.0;
    const double turn_radius = 40.0;
    
    // Track layout:
    // Section 1: Bottom straight (0 to track_length)
    // Section 2: Right turn (track_length to track_length + pi*turn_radius)
    // Section 3: Top straight (track_length + pi*turn_radius to 2*track_length + pi*turn_radius)
    // Section 4: Left turn (2*track_length + pi*turn_radius to total_length)
    
    double section1_end = track_length;
    double section2_end = track_length + M_PI * turn_radius;
    double section3_end = 2.0 * track_length + M_PI * turn_radius;
    
    if (s < section1_end) {
        // Bottom straight - traveling in +x direction
        x = -track_length/2.0 + s;
        y = -turn_radius;
        psi = 0.0;
    }
    else if (s < section2_end) {
        // Right turn - clockwise from bottom
        double arc_length = s - section1_end;
        double angle = -M_PI/2.0 + (arc_length / turn_radius);
        x = track_length/2.0 + turn_radius * cos(angle);
        y = turn_radius * sin(angle);
        psi = angle + M_PI/2.0;  // tangent to curve
    }
    else if (s < section3_end) {
        // Top straight - traveling in -x direction
        double dist_along_top = s - section2_end;
        x = track_length/2.0 - dist_along_top;
        y = turn_radius;
        psi = M_PI;
    }
    else {
        // Left turn - clockwise from top
        double arc_length = s - section3_end;
        double angle = M_PI/2.0 + (arc_length / turn_radius);
        x = -track_length/2.0 + turn_radius * cos(angle);
        y = turn_radius * sin(angle);
        psi = angle + M_PI/2.0;  // tangent to curve
    }
}

void sim_server::viz_ai_vehicle()
{
    // Get AI vehicle pose
    double ai_x, ai_y, ai_psi;
    get_track_pose(ai_path_position, ai_x, ai_y, ai_psi);
    
    // Geometry - same as player vehicle
    const double lf = 1.6;
    const double lr = 1.6;
    const double track = 1.6;
    const double wheel_radius = 0.6;
    const double wheel_thickness = 0.4;
    const double body_height = 1.0;
    const double ground_z = 0.0;
    const double body_center_z = ground_z + wheel_radius + (body_height / 2.0);
    
    // Red color for AI vehicle
    std_msgs::msg::ColorRGBA ai_body_color;
    ai_body_color.r = 1.0; ai_body_color.g = 0.0; ai_body_color.b = 0.0; ai_body_color.a = 0.95;
    std_msgs::msg::ColorRGBA wheel_color;
    wheel_color.r = 0.05; wheel_color.g = 0.05; wheel_color.b = 0.05; wheel_color.a = 1.0;
    
    std_msgs::msg::Header hdr;
    hdr.frame_id = "map";
    hdr.stamp = _node->get_clock()->now();
    
    visualization_msgs::msg::MarkerArray ma;
    
    // Body
    visualization_msgs::msg::Marker ai_body;
    ai_body.header = hdr;
    ai_body.ns = "ai_car";
    ai_body.id = 0;
    ai_body.type = visualization_msgs::msg::Marker::CUBE;
    ai_body.action = visualization_msgs::msg::Marker::ADD;
    ai_body.pose.position.x = ai_x;
    ai_body.pose.position.y = ai_y;
    ai_body.pose.position.z = body_center_z;
    ai_body.pose.orientation = quat_from_rpy(0.0, 0.0, ai_psi);
    ai_body.scale.x = lf + lr + 0.08;
    ai_body.scale.y = track + 0.06;
    ai_body.scale.z = body_height;
    ai_body.color = ai_body_color;
    ma.markers.push_back(ai_body);
    
    // Wheels
    Eigen::Vector2d w_fl(lf,  track/2.0);
    Eigen::Vector2d w_fr(lf, -track/2.0);
    Eigen::Vector2d w_rl(-lr,  track/2.0);
    Eigen::Vector2d w_rr(-lr, -track/2.0);
    std::vector<Eigen::Vector2d> wheel_positions = { w_fl, w_fr, w_rl, w_rr };
    
    Eigen::Rotation2Dd R(ai_psi);
    
    for (int i = 0; i < 4; ++i) {
        Eigen::Vector2d local = wheel_positions[i];
        Eigen::Vector2d world_xy = R * local;
        
        visualization_msgs::msg::Marker wheel;
        wheel.header = hdr;
        wheel.ns = "ai_car";
        wheel.id = 10 + i;
        wheel.type = visualization_msgs::msg::Marker::CYLINDER;
        wheel.action = visualization_msgs::msg::Marker::ADD;
        wheel.pose.position.x = ai_x + world_xy.x();
        wheel.pose.position.y = ai_y + world_xy.y();
        wheel.pose.position.z = ground_z + wheel_radius;
        wheel.pose.orientation = mult_quat(
            quat_from_rpy(0.0, 0.0, ai_psi),
            quat_from_rpy(M_PI/2.0, 0.0, 0.0)
        );
        wheel.scale.x = wheel_radius * 2.0;
        wheel.scale.y = wheel_radius * 2.0;
        wheel.scale.z = wheel_thickness;
        wheel.color = wheel_color;
        ma.markers.push_back(wheel);
    }
    
    ai_viz_pub->publish(ma);

    // Publish AI vehicle pose on /sim_server/ai_pos
    geometry_msgs::msg::PoseStamped ai_pose;
    ai_pose.header = hdr;
    ai_pose.pose.position.x = ai_x;
    ai_pose.pose.position.y = ai_y;
    ai_pose.pose.position.z = 0.0;
    ai_pose.pose.orientation = quat_from_rpy(0.0, 0.0, ai_psi);
    ai_pos_pub->publish(ai_pose);
}

void sim_server::viz_lidar()
{
    if (state_k_bike.size() != 5) {
        return;
    }

    visualization_msgs::msg::MarkerArray ma;
    
    std_msgs::msg::Header hdr;
    hdr.frame_id = "base_link";  // LiDAR in vehicle frame
    hdr.stamp = _node->get_clock()->now();
    
    // LiDAR sensor body (mounted on top of vehicle)
    const double lidar_height = 0.1;  // 10cm tall
    const double lidar_radius = 0.05; // 5cm radius
    const double mount_height = 1.6;  // mounted at 1.6m above ground (on top of vehicle body)
    
    // Sensor body color (dark gray/black)
    std_msgs::msg::ColorRGBA sensor_color;
    sensor_color.r = 0.1; sensor_color.g = 0.1; sensor_color.b = 0.1; sensor_color.a = 1.0;
    
    // Laser ray color (red/orange)
    std_msgs::msg::ColorRGBA ray_color;
    ray_color.r = 1.0; ray_color.g = 0.3; ray_color.b = 0.0; ray_color.a = 0.3;
    
    // LiDAR body
    visualization_msgs::msg::Marker lidar_body;
    lidar_body.header = hdr;
    lidar_body.ns = "lidar";
    lidar_body.id = 0;
    lidar_body.type = visualization_msgs::msg::Marker::CYLINDER;
    lidar_body.action = visualization_msgs::msg::Marker::ADD;
    lidar_body.pose.position.x = 0.0;  // center of vehicle
    lidar_body.pose.position.y = 0.0;
    lidar_body.pose.position.z = mount_height;
    lidar_body.pose.orientation.w = 1.0;
    lidar_body.scale.x = lidar_radius * 2.0;  // diameter
    lidar_body.scale.y = lidar_radius * 2.0;  // diameter
    lidar_body.scale.z = lidar_height;
    lidar_body.color = sensor_color;
    ma.markers.push_back(lidar_body);
    
    // Scan pattern visualization
    // UST-10LX: 270 degrees scan, centered forward
    // This means -135° to +135° from forward direction
    const double start_angle = -lidar_scan_angle / 2.0;  // -135 degrees
    const double end_angle = lidar_scan_angle / 2.0;     // +135 degrees
    
    // Sample rays for visualization (show every 20th ray to avoid clutter)
    const int ray_skip = 20;
    const double viz_range = lidar_max_range * 0.5;  // Show rays at half max range for visualization
    
    visualization_msgs::msg::Marker scan_rays;
    scan_rays.header = hdr;
    scan_rays.ns = "lidar";
    scan_rays.id = 1;
    scan_rays.type = visualization_msgs::msg::Marker::LINE_LIST;
    scan_rays.action = visualization_msgs::msg::Marker::ADD;
    scan_rays.scale.x = 0.04;  // line width
    scan_rays.color = ray_color;
    scan_rays.pose.orientation.w = 1.0;
    
    for (int i = 0; i < lidar_num_rays; i += ray_skip) {
        double angle = start_angle + i * lidar_angular_resolution;
        
        geometry_msgs::msg::Point p_start, p_end;
        // Start point at LiDAR sensor
        p_start.x = 0.0;
        p_start.y = 0.0;
        p_start.z = mount_height;
        
        // End point at visualization range
        p_end.x = viz_range * cos(angle);
        p_end.y = viz_range * sin(angle);
        p_end.z = mount_height;
        
        scan_rays.points.push_back(p_start);
        scan_rays.points.push_back(p_end);
    }
    ma.markers.push_back(scan_rays);
    
    // Scan coverage arc (showing the outer boundary)
    visualization_msgs::msg::Marker coverage_arc;
    coverage_arc.header = hdr;
    coverage_arc.ns = "lidar";
    coverage_arc.id = 2;
    coverage_arc.type = visualization_msgs::msg::Marker::LINE_STRIP;
    coverage_arc.action = visualization_msgs::msg::Marker::ADD;
    coverage_arc.scale.x = 0.02;  // line width
    coverage_arc.color = ray_color;
    coverage_arc.color.a = 0.6;  // more opaque for arc
    coverage_arc.pose.orientation.w = 1.0;
    
    // Draw arc at max range
    const int arc_segments = 54;  // ~5 degrees per segment
    for (int i = 0; i <= arc_segments; ++i) {
        double angle = start_angle + (lidar_scan_angle * i / arc_segments);
        geometry_msgs::msg::Point p;
        p.x = viz_range * cos(angle);
        p.y = viz_range * sin(angle);
        p.z = mount_height;
        coverage_arc.points.push_back(p);
    }
    ma.markers.push_back(coverage_arc);
    
    lidar_viz_pub->publish(ma);
}

// Ray-line segment intersection
// Returns distance to intersection, or infinity if no intersection
double sim_server::ray_line_intersection(double ray_x, double ray_y, double ray_dx, double ray_dy,
                                         double line_x1, double line_y1, double line_x2, double line_y2)
{
    // Line segment from (line_x1, line_y1) to (line_x2, line_y2)
    double line_dx = line_x2 - line_x1;
    double line_dy = line_y2 - line_y1;
    
    // Solve: ray_origin + t * ray_dir = line_start + s * line_dir
    // ray_x + t * ray_dx = line_x1 + s * line_dx
    // ray_y + t * ray_dy = line_y1 + s * line_dy
    
    double denom = ray_dx * line_dy - ray_dy * line_dx;
    if (std::abs(denom) < 1e-10) {
        return std::numeric_limits<double>::infinity(); // Parallel
    }
    
    double t = ((line_x1 - ray_x) * line_dy - (line_y1 - ray_y) * line_dx) / denom;
    double s = ((line_x1 - ray_x) * ray_dy - (line_y1 - ray_y) * ray_dx) / denom;
    
    // Check if intersection is valid (t > 0 for ray, 0 <= s <= 1 for line segment)
    if (t > 0.001 && s >= 0.0 && s <= 1.0) {
        return t;
    }
    
    return std::numeric_limits<double>::infinity();
}

// Ray-circle intersection
// Returns distance to nearest intersection, or infinity if no intersection
double sim_server::ray_circle_intersection(double ray_x, double ray_y, double ray_dx, double ray_dy,
                                           double circle_x, double circle_y, double circle_r)
{
    // Translate to circle-centered coordinates
    double ox = ray_x - circle_x;
    double oy = ray_y - circle_y;
    
    // Quadratic equation: (ox + t*dx)^2 + (oy + t*dy)^2 = r^2
    double a = ray_dx * ray_dx + ray_dy * ray_dy;
    double b = 2.0 * (ox * ray_dx + oy * ray_dy);
    double c = ox * ox + oy * oy - circle_r * circle_r;
    
    double discriminant = b * b - 4.0 * a * c;
    if (discriminant < 0.0) {
        return std::numeric_limits<double>::infinity(); // No intersection
    }
    
    double sqrt_disc = std::sqrt(discriminant);
    double t1 = (-b - sqrt_disc) / (2.0 * a);
    double t2 = (-b + sqrt_disc) / (2.0 * a);
    
    // Return nearest positive intersection
    if (t1 > 0.001) return t1;
    if (t2 > 0.001) return t2;
    
    return std::numeric_limits<double>::infinity();
}

// Ray-arc intersection (circular arc between start_angle and end_angle)
// Returns distance to nearest intersection on the arc, or infinity if no intersection
double sim_server::ray_arc_intersection(double ray_x, double ray_y, double ray_dx, double ray_dy,
                                        double circle_x, double circle_y, double circle_r,
                                        double start_angle, double end_angle)
{
    // First get intersection with full circle
    double ox = ray_x - circle_x;
    double oy = ray_y - circle_y;
    
    double a = ray_dx * ray_dx + ray_dy * ray_dy;
    double b = 2.0 * (ox * ray_dx + oy * ray_dy);
    double c = ox * ox + oy * oy - circle_r * circle_r;
    
    double discriminant = b * b - 4.0 * a * c;
    if (discriminant < 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    
    double sqrt_disc = std::sqrt(discriminant);
    double t1 = (-b - sqrt_disc) / (2.0 * a);
    double t2 = (-b + sqrt_disc) / (2.0 * a);
    
    // Check each intersection point to see if it's on the arc
    auto check_on_arc = [&](double t) -> bool {
        if (t <= 0.001) return false;
        
        // Get intersection point
        double px = ray_x + t * ray_dx;
        double py = ray_y + t * ray_dy;
        
        // Calculate angle of intersection point relative to circle center
        double angle = std::atan2(py - circle_y, px - circle_x);
        
        // Normalize angles to [0, 2π)
        auto normalize_angle = [](double a) {
            while (a < 0.0) a += 2.0 * M_PI;
            while (a >= 2.0 * M_PI) a -= 2.0 * M_PI;
            return a;
        };
        
        angle = normalize_angle(angle);
        double s_angle = normalize_angle(start_angle);
        double e_angle = normalize_angle(end_angle);
        
        // Check if angle is within arc range
        if (s_angle <= e_angle) {
            return angle >= s_angle && angle <= e_angle;
        } else {
            // Arc wraps around 0
            return angle >= s_angle || angle <= e_angle;
        }
    };
    
    // Return nearest valid intersection
    if (check_on_arc(t1)) {
        if (check_on_arc(t2) && t2 < t1) return t2;
        return t1;
    }
    if (check_on_arc(t2)) return t2;
    
    return std::numeric_limits<double>::infinity();
}

// Ray-box (rectangle) intersection
// Box is centered at (box_x, box_y) with heading box_psi
double sim_server::ray_box_intersection(double ray_x, double ray_y, double ray_dx, double ray_dy,
                                        double box_x, double box_y, double box_psi,
                                        double box_width, double box_length)
{
    // Transform ray to box-local coordinates
    double cos_psi = std::cos(-box_psi);
    double sin_psi = std::sin(-box_psi);
    
    double local_ray_x = cos_psi * (ray_x - box_x) - sin_psi * (ray_y - box_y);
    double local_ray_y = sin_psi * (ray_x - box_x) + cos_psi * (ray_y - box_y);
    double local_ray_dx = cos_psi * ray_dx - sin_psi * ray_dy;
    double local_ray_dy = sin_psi * ray_dx + cos_psi * ray_dy;
    
    // Box edges in local coordinates (axis-aligned)
    double half_length = box_length / 2.0;
    double half_width = box_width / 2.0;
    
    double min_dist = std::numeric_limits<double>::infinity();
    
    // Check 4 edges
    // Front edge: y = half_width, x in [-half_length, half_length]
    double d1 = ray_line_intersection(local_ray_x, local_ray_y, local_ray_dx, local_ray_dy,
                                      -half_length, half_width, half_length, half_width);
    min_dist = std::min(min_dist, d1);
    
    // Back edge: y = -half_width
    double d2 = ray_line_intersection(local_ray_x, local_ray_y, local_ray_dx, local_ray_dy,
                                      -half_length, -half_width, half_length, -half_width);
    min_dist = std::min(min_dist, d2);
    
    // Right edge: x = half_length
    double d3 = ray_line_intersection(local_ray_x, local_ray_y, local_ray_dx, local_ray_dy,
                                      half_length, -half_width, half_length, half_width);
    min_dist = std::min(min_dist, d3);
    
    // Left edge: x = -half_length
    double d4 = ray_line_intersection(local_ray_x, local_ray_y, local_ray_dx, local_ray_dy,
                                      -half_length, -half_width, -half_length, half_width);
    min_dist = std::min(min_dist, d4);
    
    return min_dist;
}

// Compute LiDAR range for a specific ray angle (in base_link frame)
double sim_server::compute_lidar_range(double ray_angle)
{
    // Get vehicle pose
    double vehicle_x = state_k_bike[0];
    double vehicle_y = state_k_bike[1];
    double vehicle_psi = state_k_bike[2];
    
    // Ray origin in map frame (LiDAR position)
    double ray_x = vehicle_x;
    double ray_y = vehicle_y;
    
    // Ray direction in map frame
    double world_angle = vehicle_psi + ray_angle;
    double ray_dx = std::cos(world_angle);
    double ray_dy = std::sin(world_angle);
    
    double min_range = lidar_max_range;
    
    // Track parameters
    const double track_length = 100.0;
    const double track_width = 12.0;
    const double turn_radius = 40.0;
    const double half_width = track_width / 2.0;
    
    // Check track boundaries
    // Bottom straight outer edge (y = -turn_radius - half_width)
    double d = ray_line_intersection(ray_x, ray_y, ray_dx, ray_dy,
                                     -track_length/2.0, -turn_radius - half_width,
                                     track_length/2.0, -turn_radius - half_width);
    min_range = std::min(min_range, d);
    
    // Bottom straight inner edge (y = -turn_radius + half_width)
    d = ray_line_intersection(ray_x, ray_y, ray_dx, ray_dy,
                              -track_length/2.0, -turn_radius + half_width,
                              track_length/2.0, -turn_radius + half_width);
    min_range = std::min(min_range, d);
    
    // Top straight outer edge (y = turn_radius + half_width)
    d = ray_line_intersection(ray_x, ray_y, ray_dx, ray_dy,
                              -track_length/2.0, turn_radius + half_width,
                              track_length/2.0, turn_radius + half_width);
    min_range = std::min(min_range, d);
    
    // Top straight inner edge (y = turn_radius - half_width)
    d = ray_line_intersection(ray_x, ray_y, ray_dx, ray_dy,
                              -track_length/2.0, turn_radius - half_width,
                              track_length/2.0, turn_radius - half_width);
    min_range = std::min(min_range, d);
    
    // Curved sections - outer and inner arcs (only the track portions, not full circles)
    // Right curve: from -90° to +90° (bottom to top)
    d = ray_arc_intersection(ray_x, ray_y, ray_dx, ray_dy,
                             track_length/2.0, 0.0, turn_radius + half_width,
                             -M_PI/2.0, M_PI/2.0);
    min_range = std::min(min_range, d);
    
    d = ray_arc_intersection(ray_x, ray_y, ray_dx, ray_dy,
                             track_length/2.0, 0.0, turn_radius - half_width,
                             -M_PI/2.0, M_PI/2.0);
    min_range = std::min(min_range, d);
    
    // Left curve: from +90° to +270° (top to bottom)
    d = ray_arc_intersection(ray_x, ray_y, ray_dx, ray_dy,
                             -track_length/2.0, 0.0, turn_radius + half_width,
                             M_PI/2.0, 3.0*M_PI/2.0);
    min_range = std::min(min_range, d);
    
    d = ray_arc_intersection(ray_x, ray_y, ray_dx, ray_dy,
                             -track_length/2.0, 0.0, turn_radius - half_width,
                             M_PI/2.0, 3.0*M_PI/2.0);
    min_range = std::min(min_range, d);
    
    // Check AI vehicle
    double ai_x, ai_y, ai_psi;
    get_track_pose(ai_path_position, ai_x, ai_y, ai_psi);
    
    // AI vehicle body
    const double ai_length = 3.28; // lf + lr + 0.08
    const double ai_width = 1.66;  // track + 0.06
    d = ray_box_intersection(ray_x, ray_y, ray_dx, ray_dy,
                            ai_x, ai_y, ai_psi, ai_width, ai_length);
    min_range = std::min(min_range, d);
    
    // AI vehicle wheels (4 cylinders approximated as circles)
    const double wheel_radius = 0.6;
    const double wheel_lf = 1.6;
    const double wheel_lr = 1.6;
    const double wheel_track = 1.6;
    
    // Wheel positions relative to AI vehicle
    std::vector<std::pair<double, double>> wheel_offsets = {
        {wheel_lf, wheel_track/2.0},   // FL
        {wheel_lf, -wheel_track/2.0},  // FR
        {-wheel_lr, wheel_track/2.0},  // RL
        {-wheel_lr, -wheel_track/2.0}  // RR
    };
    
    for (const auto& offset : wheel_offsets) {
        double wheel_local_x = offset.first;
        double wheel_local_y = offset.second;
        
        // Transform to world frame
        double wheel_world_x = ai_x + wheel_local_x * std::cos(ai_psi) - wheel_local_y * std::sin(ai_psi);
        double wheel_world_y = ai_y + wheel_local_x * std::sin(ai_psi) + wheel_local_y * std::cos(ai_psi);
        
        d = ray_circle_intersection(ray_x, ray_y, ray_dx, ray_dy,
                                   wheel_world_x, wheel_world_y, wheel_radius);
        min_range = std::min(min_range, d);
    }
    
    return min_range;
}

void sim_server::publish_lidar_scan(const rclcpp::Time& timestamp)
{
    sensor_msgs::msg::LaserScan scan;
    
    // Header
    scan.header.stamp = timestamp;
    scan.header.frame_id = "base_link";  // LiDAR frame
    
    // LaserScan parameters for UST-10LX
    // 270 degree scan from -135° to +135° (centered forward)
    scan.angle_min = -lidar_scan_angle / 2.0;  // -135 degrees in radians
    scan.angle_max = lidar_scan_angle / 2.0;   // +135 degrees in radians
    scan.angle_increment = lidar_angular_resolution;  // 0.25 degrees in radians
    scan.time_increment = 1.0 / (lidar_frequency * lidar_num_rays);  // time between measurements
    scan.scan_time = 1.0 / lidar_frequency;  // 1/40 = 0.025 seconds
    scan.range_min = 0.1;   // minimum range (typically 10cm for safety)
    scan.range_max = lidar_max_range;  // 30 meters
    
    // Initialize ranges array
    scan.ranges.resize(lidar_num_rays);
    scan.intensities.resize(lidar_num_rays);
    
    // Compute actual ranges using ray-tracing
    const double start_angle = -lidar_scan_angle / 2.0;
    
    for (int i = 0; i < lidar_num_rays; ++i) {
        double ray_angle = start_angle + i * lidar_angular_resolution;
        double range = compute_lidar_range(ray_angle);
        
        // Clamp to valid range
        if (range < scan.range_min) {
            range = scan.range_min;
        } else if (range > scan.range_max) {
            range = scan.range_max;
        }
        
        scan.ranges[i] = range;
        
        // Intensity based on range (closer = higher intensity)
        if (range < scan.range_max) {
            scan.intensities[i] = 200.0 * (1.0 - range / scan.range_max);
        } else {
            scan.intensities[i] = 50.0; // Low intensity for max range
        }
    }
    
    lidar_scan_pub->publish(scan);
}

geometry_msgs::msg::Quaternion sim_server::mult_quat(
    const geometry_msgs::msg::Quaternion &a, 
    const geometry_msgs::msg::Quaternion &b)
{
    geometry_msgs::msg::Quaternion r;
    r.w = a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z;
    r.x = a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y;
    r.y = a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x;
    r.z = a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w;
    return r;
}

void sim_server::rcCallback(const sensor_msgs::msg::Joy::ConstPtr msg)
{
    acc_input_bike <<
        msg->axes[3] * 10.0, 
        msg->axes[2] * 20.0 / 180.0 * M_PI;
    
}
