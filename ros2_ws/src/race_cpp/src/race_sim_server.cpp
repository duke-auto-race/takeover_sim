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

    track_pub = _node->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/sim_server/track_marker",
        10
    );

    ai_viz_pub = _node->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/sim_server/ai_marker",
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

    // Broadcast TF transform
    geometry_msgs::msg::TransformStamped transform;
    transform.header.stamp = _node->get_clock()->now();
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
    camera_transform.header.stamp = _node->get_clock()->now();
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
    
    // Lane markings (center dashed line)
    int num_dashes = 40;
    double dash_length = (track_length + M_PI * turn_radius) / num_dashes;
    
    // Dashes on bottom straight
    for (int i = 0; i < int(track_length / dash_length / 2); ++i) {
        if (i % 2 == 0) continue;  // skip every other for dashed effect
        visualization_msgs::msg::Marker dash;
        dash.header = hdr;
        dash.ns = "track";
        dash.id = marker_id++;
        dash.type = visualization_msgs::msg::Marker::CUBE;
        dash.action = visualization_msgs::msg::Marker::ADD;
        dash.pose.position.x = -track_length/2.0 + i * dash_length;
        dash.pose.position.y = -turn_radius;
        dash.pose.position.z = 0.01;
        dash.pose.orientation.w = 1.0;
        dash.scale.x = dash_length * 0.8;
        dash.scale.y = 0.3;
        dash.scale.z = road_thickness;
        dash.color = white_color;
        track_markers.markers.push_back(dash);
    }
    
    track_pub->publish(track_markers);
    RCLCPP_INFO(_node->get_logger(), "Track visualization published with %zu markers", track_markers.markers.size());
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
