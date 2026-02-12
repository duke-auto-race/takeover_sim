#!/usr/bin/env python3
import threading
import struct
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

JS_EVENT_FORMAT = "IhBB"
JS_EVENT_SIZE = struct.calcsize(JS_EVENT_FORMAT)
JS_EVENT_BUTTON = 0x01
JS_EVENT_AXIS = 0x02
JS_EVENT_INIT = 0x80

DEVICE_DEFAULT = "/dev/input/js0"
PUBLISH_HZ_DEFAULT = 50.0  # Hz
TOPIC_DEFAULT = "/channels"  # publishes [ch0, ch2]


class JoystickReader(threading.Thread):
    def __init__(self, device=DEVICE_DEFAULT):
        super().__init__(daemon=True)
        self.device = device
        self._stop = threading.Event()
        self.axes = {}      # axis_index -> raw int16
        self.buttons = {}   # button_index -> int
        self._lock = threading.Lock()

    def run(self):
        try:
            with open(self.device, "rb") as f:
                while not self._stop.is_set():
                    data = f.read(JS_EVENT_SIZE)
                    if len(data) != JS_EVENT_SIZE:
                        continue
                    _, value, evtype, number = struct.unpack(JS_EVENT_FORMAT, data)
                    base_type = evtype & (~JS_EVENT_INIT)
                    with self._lock:
                        if base_type == JS_EVENT_AXIS:
                            self.axes[int(number)] = int(value)
                        elif base_type == JS_EVENT_BUTTON:
                            self.buttons[int(number)] = int(value != 0)
        except FileNotFoundError:
            raise
        except Exception:
            raise

    def stop(self):
        self._stop.set()

    def get_axis(self, idx):
        with self._lock:
            return self.axes.get(idx, 0)

    def get_axis_normalized(self, idx):
        raw = self.get_axis(idx)
        if raw <= -32768:
            return -1.0
        return float(raw) / 32767.0


class JsPublisherNode(Node):
    def __init__(self):
        super().__init__("js_channel_publisher_array")
        # params
        self.declare_parameter("device", DEVICE_DEFAULT)
        self.declare_parameter("rate_hz", float(PUBLISH_HZ_DEFAULT))
        self.declare_parameter("topic", TOPIC_DEFAULT)

        device = self.get_parameter("device").get_parameter_value().string_value
        rate_hz = float(self.get_parameter("rate_hz").get_parameter_value().double_value)
        topic = self.get_parameter("topic").get_parameter_value().string_value

        self.pub = self.create_publisher(Float32MultiArray, topic, 10)

        self.get_logger().info(
            f"Starting joystick reader for '{device}', publishing {topic} at {rate_hz:.1f} Hz"
        )

        # start joystick reader thread
        try:
            self.js = JoystickReader(device=device)
            self.js.start()
        except FileNotFoundError:
            self.get_logger().error(f"Joystick device not found: {device}")
            raise
        except Exception as e:
            self.get_logger().error(f"Failed to start joystick reader: {e}")
            raise

        # timer
        period = 1.0 / max(rate_hz, 1e-6)
        self.timer = self.create_timer(period, self.timer_callback)

    def timer_callback(self):
        # pack channel 0 and channel 2 into a Float32MultiArray
        val0 = self.js.get_axis_normalized(0)
        val2 = self.js.get_axis_normalized(2)
        
        val0 = -val0
        val2 = (-val2+1)/2
        # print("hi=")
        
        msg = Float32MultiArray()
        msg.data = [float(val0), float(val2)]
        self.pub.publish(msg)

    def destroy_node(self):
        try:
            self.js.stop()
            self.js.join(timeout=1.0)
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = JsPublisherNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        if node is not None:
            node.get_logger().error(f"Exception in node: {e}")
        else:
            print(f"Exception: {e}")
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                pass
        rclpy.shutdown()


if __name__ == "__main__":
    main()