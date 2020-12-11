import rospy
from tf import transformations as t
import tf
import tf2_ros
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu


class Pioneer_Pose(object):
	imu_pub = rospy.Publisher('/imu', Imu, queue_size=1)
	imu_1_pub = rospy.Publisher('/imu1', Imu, queue_size=1)
	imu_2_pub = rospy.Publisher('/imu2', Imu, queue_size=1)
	imu_3_pub = rospy.Publisher('/imu3', Imu, queue_size=1)
	# pub_map_pose = rospy.Publisher('/map_pose_2',PoseWithCovarianceStamped,queue_size=1)

	def __init__(self):
		# self.listener = tf.TransformListener(rospy.Time(0))

		# rospy.Subscriber('/initialpose', PoseWithCovarianceStamped, self.cb_initial, queue_size=1, buff_size=2**10)
		rospy.Subscriber('/RosAria/pose', Odometry, self.cb_odom, queue_size=1, buff_size=2**10)
		# rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.cb_global, queue_size=1, buff_size=2**10)
		# rospy.Subscriber('/map_pose', Odometry, self.cb_map_pose, queue_size=1, buff_size=2**10)
		rospy.Subscriber('/mynteye/imu/data_raw', Imu, self.cb_imu, queue_size=1, buff_size=2**10)
		rospy.Subscriber('/mynteye_1/imu/data_raw', Imu, self.cb_imu_1, queue_size=1, buff_size=2**10)
		rospy.Subscriber('/mynteye_2/imu/data_raw', Imu, self.cb_imu_2, queue_size=1, buff_size=2**10)
		rospy.Subscriber('/mynteye_3/imu/data_raw', Imu, self.cb_imu_3, queue_size=1, buff_size=2**10)

		
		# self.pub_map_pose = rospy.Publisher('/map_pose', PoseWithCovarianceStamped, queue_size=1)
		self.pub_odom = rospy.Publisher('/Pioneer_odom', Odometry, queue_size=1)

	def initialize_map_odom_tf(self,):
		map_to_odom = TransformStamped()

		map_to_odom.header.stamp = rospy.Time.now()
		map_to_odom.header.frame_id = 'map'
		map_to_odom.child_frame_id = 'odom'
		map_to_odom.transform.translation.x, map_to_odom.transform.translation.y, map_to_odom.transform.translation.z  = 0,0,0
		map_to_odom.transform.rotation.x, map_to_odom.transform.rotation.y, map_to_odom.transform.rotation.z, map_to_odom.transform.rotation.w = 0,0,0,1

		br = tf2_ros.StaticTransformBroadcaster()
		br.sendTransform(map_to_odom)

	def cb_odom(self, msg):

		msg.header.frame_id = 'odom'
		msg.child_frame_id = 'base_link'

		self.pub_odom.publish(msg)

	def cb_global(self,msg):

		self.listener.waitForTransform("map", "odom", msg.header.stamp,  rospy.Duration(0.1))
		self.listener.waitForTransform("odom", "base_link", msg.header.stamp,  rospy.Duration(0.1))
		print ('mo',self.listener.lookupTransform("map", "odom" , msg.header.stamp))
		print ('ob',self.listener.lookupTransform("odom", "base_link" , msg.header.stamp))
		print ('mb',self.listener.lookupTransform("map", "base_link" , msg.header.stamp))
		(trans, rot) = self.listener.lookupTransform("map", "base_link" , msg.header.stamp)
		
		map_pose = PoseWithCovarianceStamped()
		map_pose.header.stamp = msg.header.stamp
		map_pose.header.frame_id = "map"
		map_pose.pose.pose.position.x, map_pose.pose.pose.position.y, map_pose.pose.pose.position.z = trans
		map_pose.pose.pose.orientation.x, map_pose.pose.pose.orientation.y, map_pose.pose.pose.orientation.z, map_pose.pose.pose.orientation.w = rot

		self.pub_map_pose.publish(map_pose)


	def cb_initial(self,msg):

		trans = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
		rot = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
		print ("Re-Initializing pose to: ", trans, rot)

		# Calculate BASE_LINK to MAP transform 
		transform = t.concatenate_matrices(t.translation_matrix(trans), t.quaternion_matrix(rot))
		inversed_transform = t.inverse_matrix(transform)

		inv_trans = t.translation_from_matrix(inversed_transform)
		inv_rot = t.quaternion_from_matrix(inversed_transform)


		base_to_map_pose = PoseStamped()
		base_to_map_pose.header.stamp = msg.header.stamp
		base_to_map_pose.header.frame_id = "base_link"
		base_to_map_pose.pose.position.x, base_to_map_pose.pose.position.y, base_to_map_pose.pose.position.z = inv_trans
		base_to_map_pose.pose.orientation.x, base_to_map_pose.pose.orientation.y, base_to_map_pose.pose.orientation.z, base_to_map_pose.pose.orientation.w = inv_rot

		self.listener.waitForTransform("odom", "base_link",msg.header.stamp,  rospy.Duration(0.5))

		map_odom = self.listener.transformPose('odom',base_to_map_pose)

		# Calculate MAP to ODOM transform
		trans_odom = [map_odom.pose.position.x, map_odom.pose.position.y, map_odom.pose.position.z]
		rot_odom = [map_odom.pose.orientation.x, map_odom.pose.orientation.y, map_odom.pose.orientation.z, map_odom.pose.orientation.w]
		
		transform_odom = t.concatenate_matrices(t.translation_matrix(trans_odom), t.quaternion_matrix(rot_odom))
		inversed_transform_odom = t.inverse_matrix(transform_odom)

		inv_trans_odom = t.translation_from_matrix(inversed_transform_odom)
		inv_rot_odom = t.quaternion_from_matrix(inversed_transform_odom)


		map_to_odom = TransformStamped()

		map_to_odom.header.stamp = rospy.Time.now()
		map_to_odom.header.frame_id = 'map'
		map_to_odom.child_frame_id = 'odom'
		map_to_odom.transform.translation.x, map_to_odom.transform.translation.y, map_to_odom.transform.translation.z  = inv_trans_odom
		map_to_odom.transform.rotation.x, map_to_odom.transform.rotation.y, map_to_odom.transform.rotation.z, map_to_odom.transform.rotation.w = inv_rot_odom

		br = tf2_ros.StaticTransformBroadcaster()
		br.sendTransform(map_to_odom)


	def cb_imu(self, msg):
		gyro = msg.angular_velocity
		acc = msg.linear_acceleration

		# Create new Imu Message

		imu = Imu()
		imu.header.frame_id = "mynteye_link_frame"
		imu.header.stamp = msg.header.stamp
		imu.orientation = msg.orientation

		imu.angular_velocity.x = -gyro.x
		imu.angular_velocity.y = gyro.z
		imu.angular_velocity.z = gyro.y

		imu.linear_acceleration.x = -acc.x
		imu.linear_acceleration.y = acc.z
		imu.linear_acceleration.z = acc.y

		self.imu_pub.publish(imu)

	def cb_imu_1(self, msg):
		gyro = msg.angular_velocity
		acc = msg.linear_acceleration

		# Create new Imu Message

		imu = Imu()
		imu.header.frame_id = "mynteye_1_link"
		imu.header.stamp = msg.header.stamp
		imu.orientation = msg.orientation

		imu.angular_velocity.x = gyro.z
		imu.angular_velocity.y = -gyro.y
		imu.angular_velocity.z = gyro.x

		imu.linear_acceleration.x = acc.z
		imu.linear_acceleration.y = -acc.y
		imu.linear_acceleration.z = acc.x

		self.imu_1_pub.publish(imu)

	def cb_imu_2(self, msg):
		gyro = msg.angular_velocity
		acc = msg.linear_acceleration

		# Create new Imu Message

		imu = Imu()
		imu.header.frame_id = "mynteye_2_link"
		imu.header.stamp = msg.header.stamp
		imu.orientation = msg.orientation

		imu.angular_velocity.x = gyro.z
		imu.angular_velocity.y = -gyro.y
		imu.angular_velocity.z = gyro.x

		imu.linear_acceleration.x = acc.z
		imu.linear_acceleration.y = -acc.y
		imu.linear_acceleration.z = acc.x

		self.imu_2_pub.publish(imu)

	def cb_imu_3(self, msg):
		gyro = msg.angular_velocity
		acc = msg.linear_acceleration

		# Create new Imu Message

		imu = Imu()
		imu.header.frame_id = "mynteye_3_link"
		imu.header.stamp = msg.header.stamp
		imu.orientation = msg.orientation

		imu.angular_velocity.x = gyro.z
		imu.angular_velocity.y = -gyro.y
		imu.angular_velocity.z = gyro.x

		imu.linear_acceleration.x = acc.z
		imu.linear_acceleration.y = -acc.y
		imu.linear_acceleration.z = acc.x

		self.imu_3_pub.publish(imu)




if __name__ == '__main__':

	rospy.init_node("Pioneer_pose")

	P = Pioneer_Pose()

	# P.initialize_map_odom_tf()

	rospy.spin()


