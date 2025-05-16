#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <image_geometry/pinhole_camera_model.h>
#include <tf/transform_listener.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/opencv.hpp>
#include <boost/foreach.hpp>

class FrameDrawer
{
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::CameraSubscriber sub_;
  image_transport::Publisher pub_;
  tf::TransformListener tf_listener_;
  image_geometry::PinholeCameraModel cam_model_;
  std::vector<std::string> frame_ids_;

public:
  FrameDrawer(const std::vector<std::string>& frame_ids)
    : it_(nh_), frame_ids_(frame_ids)
  {
    std::string image_topic = nh_.resolveName("image");
    sub_ = it_.subscribeCamera(image_topic, 1, &FrameDrawer::imageCb, this);
    pub_ = it_.advertise("image_out", 1);
  }

  void imageCb(const sensor_msgs::ImageConstPtr& image_msg,
               const sensor_msgs::CameraInfoConstPtr& info_msg)
  {
    cv::Mat image;
    cv_bridge::CvImagePtr input_bridge;
    try {
      input_bridge = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
      image = input_bridge->image;
    }
    catch (cv_bridge::Exception& ex){
      ROS_ERROR("[draw_frames] Failed to convert image");
      return;
    }

    cam_model_.fromCameraInfo(info_msg);

    // Print image bounds
    // ROS_INFO("Image bounds: width = %d, height = %d", image.cols, image.rows);

    // Draw ground truth point in the middle of the image
    // cv::Point2d image_center(image.cols / 2, image.rows / 2);
    // cv::circle(image, image_center, 5, cv::Scalar(255, 255, 255), -1); // White circle
    // cv::putText(image, "center", cv::Point(image_center.x + 10, image_center.y),
    //             cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

    BOOST_FOREACH(const std::string& frame_id, frame_ids_) {
      tf::StampedTransform transform;
      try {
        ros::Time acquisition_time = info_msg->header.stamp;
        ros::Duration timeout(1.0 / 5);
        tf_listener_.waitForTransform(cam_model_.tfFrame(), frame_id,
                                      acquisition_time, timeout);
        tf_listener_.lookupTransform(cam_model_.tfFrame(), frame_id,
                                     acquisition_time, transform);
      }
      catch (tf::TransformException& ex) {
        ROS_WARN("[draw_frames] TF exception:\n%s", ex.what());
        return;
      }

      // Get the origin point
      tf::Point origin = transform.getOrigin();
      cv::Point3d origin_cv(origin.x(), origin.y(), origin.z());
      cv::Point2d origin_uv = cam_model_.project3dToPixel(origin_cv);

      // Print origin coordinates
      // ROS_INFO("Frame %s origin: 3D (%.2f, %.2f, %.2f) -> 2D (%.2f, %.2f)", 
      //          frame_id.c_str(), origin.x(), origin.y(), origin.z(), origin_uv.x, origin_uv.y);

      double axis_length = 0.1; 

      tf::Vector3 x_axis = transform * tf::Vector3(axis_length, 0, 0);
      tf::Vector3 y_axis = transform * tf::Vector3(0, axis_length, 0);
      tf::Vector3 z_axis = transform * tf::Vector3(0, 0, axis_length);

      // Project axis end points
      cv::Point2d x_uv = cam_model_.project3dToPixel(cv::Point3d(x_axis.x(), x_axis.y(), x_axis.z()));
      cv::Point2d y_uv = cam_model_.project3dToPixel(cv::Point3d(y_axis.x(), y_axis.y(), y_axis.z()));
      cv::Point2d z_uv = cam_model_.project3dToPixel(cv::Point3d(z_axis.x(), z_axis.y(), z_axis.z()));

      // // Print axis end point coordinates
      // ROS_INFO("Frame %s X-axis end: 3D (%.2f, %.2f, %.2f) -> 2D (%.2f, %.2f)", 
      //          frame_id.c_str(), x_axis.x(), x_axis.y(), x_axis.z(), x_uv.x, x_uv.y);
      // ROS_INFO("Frame %s Y-axis end: 3D (%.2f, %.2f, %.2f) -> 2D (%.2f, %.2f)", 
      //          frame_id.c_str(), y_axis.x(), y_axis.y(), y_axis.z(), y_uv.x, y_uv.y);
      // ROS_INFO("Frame %s Z-axis end: 3D (%.2f, %.2f, %.2f) -> 2D (%.2f, %.2f)", 
      //          frame_id.c_str(), z_axis.x(), z_axis.y(), z_axis.z(), z_uv.x, z_uv.y);

      // Draw axes
      cv::line(image, origin_uv, x_uv, cv::Scalar(0, 0, 255), 2); // X-axis (Red)
      cv::line(image, origin_uv, y_uv, cv::Scalar(0, 255, 0), 2); // Y-axis (Green)
      cv::line(image, origin_uv, z_uv, cv::Scalar(255, 0, 0), 2); // Z-axis (Blue)

      // Draw the frame ID with an outline for better contrast
      std::string text = frame_id;
      int font_face = cv::FONT_HERSHEY_SIMPLEX;
      double font_scale = 0.5;
      int thickness = 1;

      int baseline = 0;
      cv::Size text_size = cv::getTextSize(text, font_face, font_scale, thickness, &baseline);
      cv::Point text_origin(origin_uv.x - text_size.width / 2, origin_uv.y - text_size.height - 5);

      // Draw outline (thicker, in black)
      cv::putText(image, text, text_origin, font_face, font_scale, cv::Scalar(0, 0, 0), thickness + 2);  // Outline

      // Draw the frame ID in a contrasting color
      cv::putText(image, text, text_origin, font_face, font_scale, cv::Scalar(50, 205, 50), thickness);  // Lime green text
     }

    pub_.publish(input_bridge->toImageMsg());
  }
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "draw_frames");
  std::vector<std::string> frame_ids(argv + 1, argv + argc);
  FrameDrawer drawer(frame_ids);
  ros::spin();
}