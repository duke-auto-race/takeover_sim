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

        # Step size for incremental control per second
        self.step_per_sec = 2.0  # value per second

        self.get_logger().info('RC Teleop Node Started')
        self.get_logger().info('Use Arrow Keys:')
        self.get_logger().info('  Up/Down: Channel 1 (-1 to 1)')
        self.get_logger().info('  Left/Right: Channel 2 (-1 to 1)')
        self.get_logger().info('  Space: Reset both to 0')
        self.get_logger().info('  q: Quit')

        # Save terminal settings
        self.settings = termios.tcgetattr(sys.stdin)

        # Track last input time
        self.last_input_time = None
        self.active_key = None
        
    def get_key(self, timeout=0.1):
        """Get a single keypress from stdin with timeout (seconds)"""
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        key = None
        if rlist:
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
        import time
        try:
            self.last_input_time = time.time()
            self.active_key = None
            while rclpy.ok():
                now = time.time()
                key = self.get_key(timeout=0.1)

                # If key is pressed
                if key:
                    self.last_input_time = now
                    # Check for escape sequences (arrow keys)
                    if key == '\x1b':  # ESC
                        next1 = sys.stdin.read(1)
                        if next1 == '[':
                            next2 = sys.stdin.read(1)
                            if next2 == 'A':  # Up arrow
                                self.active_key = 'up'
                            elif next2 == 'B':  # Down arrow
                                self.active_key = 'down'
                            elif next2 == 'C':  # Right arrow
                                self.active_key = 'right'
                            elif next2 == 'D':  # Left arrow
                                self.active_key = 'left'
                            else:
                                self.active_key = None
                        else:
                            self.active_key = None
                    elif key == ' ':  # Space - reset
                        self.channel1_value = 0.0
                        self.channel2_value = 0.0
                        self.publish_values()
                        self.active_key = None
                    elif key == 'q' or key == '\x03':  # q or Ctrl+C
                        break
                    else:
                        self.active_key = None
                else:
                    # No key pressed in this interval
                    pass

                # If active key is held, update values based on time held
                if self.active_key:
                    dt = 0.1  # seconds per loop
                    step = self.step_per_sec * dt
                    if self.active_key == 'up':
                        self.channel1_value = self.clamp(self.channel1_value + step)
                    elif self.active_key == 'down':
                        self.channel1_value = self.clamp(self.channel1_value - step)
                    elif self.active_key == 'right':
                        self.channel2_value = self.clamp(self.channel2_value - step)
                    elif self.active_key == 'left':
                        self.channel2_value = self.clamp(self.channel2_value + step)
                    self.publish_values()

                # If no input for over 100 ms, reset values
                if (now - self.last_input_time) > 0.1:
                    if self.channel1_value != 0.0 or self.channel2_value != 0.0:
                        self.channel1_value = 0.0
                        self.channel2_value = 0.0
                        self.publish_values()
                    self.active_key = None

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
