import numpy as np
import cv2

class FlowCalculator:
    def __init__(self):
        self.nh = rospy.init_node('flow_calculator')
        self.it = ImageTransport(self.nh)
        self.image_sub = self.it.subscribe("/usb_cam/image_raw", 1, self.img_cb)
        self.pose_pub = rospy.Publisher("/optical_flow/pose", Pose2D, queue_size=1)
        self.twist_pub = rospy.Publisher("/optical_flow/twist", Twist, queue_size=1)
        self.bridge = CvBridge()
        
        # Rest of the initialization code...

    def img_cb(self, input):
        # Rest of the image callback function code...

        # Convert optical flow results to Pose2D and publish
        pose_out = Pose2D()
        pose_out.x = accumulated_motion[1]
        pose_out.y = 0
        pose_out.theta = accumulated_motion[0] / (2 * np.pi * 0.4485) * 360 * 1.57
        self.pose_pub.publish(pose_out)

        # Convert optical flow results to Twist and publish
        t_curr = rospy.Time.now().to_sec()
        twist_out = Twist()
        twist_out.linear.x = accumulated_motion[1] / (t_curr - t_prev)
        twist_out.angular.z = accumulated_motion[0] / (2 * np.pi * 0.4485) * 2 * np.pi * 1.57 / (t_curr - t_prev)

        # Apply filtering and publish the Twist
        filter[filter_count] = twist_out
        twist_out.linear.x = 0
        twist_out.linear.y = 0
        twist_out.linear.z = 0
        twist_out.angular.x = 0
        twist_out.angular.y = 0
        twist_out.angular.z = 0
        for i in range(5):
            twist_out.linear.x += filter[i].linear.x / 5
            twist_out.linear.y += filter[i].linear.y / 5
            twist_out.linear.z += filter[i].linear.z / 5
            twist_out.angular.x += filter[i].angular.x / 5
            twist_out.angular.y += filter[i].angular.y / 5
            twist_out.angular.z += filter[i].angular.z / 5
        self.twist_pub.publish(twist_out)

        filter_count = (filter_count + 1) % 5
        t_prev = t_curr

        # Rest of the image processing code...

if __name__ == "__main__":
    flow_calculator = FlowCalculator()
    rospy.spin()
