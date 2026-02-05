#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import sys
import tty
import termios
import select

class RCTeleopNode(Node):
    def __init__(self):
        super().__init__('rc_teleop_node')
        
        self.rc_pub = self.create_publisher(Float64MultiArray, '/rc/virtual', 10)
        
        self.channel1_value = 0.0  
        self.channel2_value = 0.0 
        
        # Step size for incremental control
        self.step = 0.1
        
        self.get_logger().info('RC Teleop Node Started')
        self.get_logger().info('Use Arrow Keys:')
        self.get_logger().info('  Up/Down: Channel 1 (-1 to 1)')
        self.get_logger().info('  Left/Right: Channel 2 (-1 to 1)')
        self.get_logger().info('  Space: Reset both to 0')
        self.get_logger().info('  q: Quit')
        
        # Save terminal settings
        self.settings = termios.tcgetattr(sys.stdin)
        
    def get_key(self):
        """Get a single keypress from stdin"""
        tty.setraw(sys.stdin.fileno())
        select.select([sys.stdin], [], [], 0)
        key = sys.stdin.read(1)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key
    
    def clamp(self, value, min_val=-1.0, max_val=1.0):
        """Clamp value between min and max"""
        return max(min_val, min(max_val, value))
    
    def publish_values(self):
        """Publish current channel values"""
        msg = Float64MultiArray()
        msg.data = [self.channel1_value, self.channel2_value]
        self.rc_pub.publish(msg)
        
        self.get_logger().info(f'Ch1: {self.channel1_value:.2f}, Ch2: {self.channel2_value:.2f}')
    
    def run(self):
        """Main control loop"""
        try:
            while rclpy.ok():
                key = self.get_key()
                
                # Check for escape sequences (arrow keys)
                if key == '\x1b':  # ESC
                    next1 = sys.stdin.read(1)
                    if next1 == '[':
                        next2 = sys.stdin.read(1)
                        if next2 == 'A':  # Up arrow
                            self.channel1_value = self.clamp(self.channel1_value + self.step)
                            self.publish_values()
                        elif next2 == 'B':  # Down arrow
                            self.channel1_value = self.clamp(self.channel1_value - self.step)
                            self.publish_values()
                        elif next2 == 'C':  # Right arrow
                            self.channel2_value = self.clamp(self.channel2_value - self.step)
                            self.publish_values()
                        elif next2 == 'D':  # Left arrow
                            self.channel2_value = self.clamp(self.channel2_value + self.step)
                            self.publish_values()
                elif key == ' ':  # Space - reset
                    self.channel1_value = 0.0
                    self.channel2_value = 0.0
                    self.publish_values()
                elif key == 'q' or key == '\x03':  # q or Ctrl+C
                    break
                    
        except Exception as e:
            self.get_logger().error(f'Error: {str(e)}')
        finally:
            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
            # Send zero values before exiting
            self.channel1_value = 0.0
            self.channel2_value = 0.0
            self.publish_values()

def main(args=None):
    rclpy.init(args=args)
    node = RCTeleopNode()
    
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
