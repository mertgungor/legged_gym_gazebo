#include <chrono>
#include <memory>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/common.h>
#include <pcl/cloud_iterator.h>

#include "rclcpp/rclcpp.hpp"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include "sensor_msgs/msg/point_cloud2.hpp"

#include <tf2_eigen/tf2_eigen.h>

#include <stdlib.h>

using namespace std::chrono_literals;

class PointcloudTransformer : public rclcpp::Node
{
    public:

    PointcloudTransformer() : Node("tf2_listener")
    {
        tf_buffer_   = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
        transform_    = std::make_shared<geometry_msgs::msg::TransformStamped>();

        timer_ = this->create_wall_timer(100ms, std::bind(&PointcloudTransformer::timer_callback, this));

        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "velodyne_points", 10, std::bind(&PointcloudTransformer::pointcloud_callback, this, std::placeholders::_1));
        pointcloud_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("transformed_pointcloud", 10);
  
    }

    private:

    void timer_callback(){
        try 
        {
            (*transform_) = tf_buffer_->lookupTransform("base_link", "velodyne", tf2::TimePointZero);
            // RCLCPP_INFO(this->get_logger(), "Transform: %f %f %f", transform->transform.translation.x, transform->transform.translation.y, transform->transform.translation.z);
        } catch (tf2::TransformException &ex) {
            RCLCPP_WARN(this->get_logger(), "Failure %s", ex.what());
        }
    }

    void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) const
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_temp(new pcl::PointCloud<pcl::PointXYZ>);


        // pcl::fromROSMsg(*msg, *cloud_temp);

        //Tranform point cloud from velodyne frame to base_link frame

        sensor_msgs::msg::PointCloud2 transformed_pointcloud;
        tf2::doTransform(*msg, transformed_pointcloud, *transform_);


        // pointcloud_publisher_->publish(*msg);

        pcl::fromROSMsg(*msg, *cloud);
        // printf("%f", (*cloud).points[0].x);
        
        // Finds the highest point in the pointcloud for every 10cm square


        int x{0};
        int y{0};

        float min_x{0.0};
        float min_y{0.0};

        float max_x{0.0};
        float max_y{0.0};

        std::vector<std::vector<float>> highest_points(11, std::vector<float>(17, -5.0)); // 11x17 grid initialized to -5m

        for(pcl::PointCloud<pcl::PointXYZ>::iterator it = cloud->begin(); it != cloud->end(); ++it)
        {

            printf("%f %f %f\n", it->x, it->y, it->z);
            
        }


        exit(-1);

        // for(int i = 0; i < 11; i++)
        // {
        //     for(int j = 0; j < 17; j++)
        //     {
        //         std::cout << "X: " << ((float)i - 5) / 10 << " Y: " << ((float)j - 8 ) / 10 << " ";
        //         std::cout << highest_points[i][j] << std::endl;
        //     }
        //     std::cout << "-------------------" << std::endl;
        // }
        // std::cout << "===========" << std::endl << std::endl << std::endl;



        // std::vector<std::vector<int>> highest_points(11, std::vector<int>(17, -5)); // 11x17 grid initialized to -5m

        // for(int i = 0; i < 11; i++)
        // {
        //     for(int j = 0; j < 17; j++)
        //     {
        //         std::cout << highest_points[i][j] << std::endl;
        //     }
        // }

        // RCLCPP_INFO(this->get_logger(), "Pointcloud received");
        // RCLCPP_INFO(this->get_logger(), "X min: %f", x_min);
        // RCLCPP_INFO(this->get_logger(), "X max: %f", x_max);

    }

    rclcpp::TimerBase::SharedPtr timer_{nullptr};
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_publisher_;

    std::shared_ptr<tf2_ros::TransformListener> tf_listener_{nullptr};
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<geometry_msgs::msg::TransformStamped> transform_{nullptr};
    std::string target_frame_;
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PointcloudTransformer>());
    rclcpp::shutdown();
    return 0;
}