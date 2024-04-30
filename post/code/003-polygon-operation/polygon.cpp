#include <iostream>
#include <string>
#include <deque>
#include <vector>   // To contain points
#include <random>   // Random generate points
#include <stdexcept>// for std::runtime_error
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/geometries.hpp> // Points, lines, polygons, etc.
#include <boost/geometry/geometries/point_xy.hpp>   // 2 D point point_xy
#include <boost/geometry/geometries/polygon.hpp>    // Polygon geometry, sequence of points for a closed loop

namespace bg = boost::geometry;
// Create type aliases
using Point = bg::model::d2::point_xy<double>;
using Points = std::vector<Point>;
using Polygon = bg::model::polygon<Point>;

Polygon getOverlappingPolygon(const Polygon& polygon_1, const Polygon& polygon_2)
{
    std::deque<Polygon> overlapping_polygon;    // The overlapping polygon number can more than one
    bg::intersection(polygon_1, polygon_2, overlapping_polygon);
    return overlapping_polygon.front();
}

bool checkPolygonIntersection(const Polygon& polygon_1, const Polygon& polygon_2)
{           
    return bg::covered_by(polygon_1, polygon_2) || bg::overlaps(polygon_1, polygon_2);
}

void printPoints(const Points& points) 
{
    std::cout << "Generated Points:" << std::endl;
    for (const auto& point : points) {
        std::cout << "(" << bg::get<0>(point) << ", " << bg::get<1>(point) << ")" << std::endl;
    }
}

Points generateRandomPoints(int num_points, double min_x, double max_x, double min_y, double max_y)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist_x(min_x, max_x);
    std::uniform_real_distribution<double> dist_y(min_y, max_y);
    Points points;
    for (int i = 0; i < num_points; ++i)
    {
        points.push_back(Point(dist_x(gen), dist_y(gen)));
    }
    points.push_back(points.front());   // To close the loop, otherwise polygon is invalid
    return points;
}

Polygon generatePolygon(const Points& points)
{
    Polygon polygon;
    bg::assign_points(polygon, points);
    return polygon;
}

void printPolygon(const Polygon polygon, const std::string& prefix = "")
{
    std::cout << "Polygon " << prefix << " points are: ";
    for (const auto& point : polygon.outer()) 
    {
        std::cout << "(" << boost::geometry::get<0>(point) << ", " << boost::geometry::get<1>(point) << ") ";
    }
    std::cout << "\t Area is: " << bg::area(polygon) << std::endl;
}

Polygon generateRandomPolygon(int num_points = 4, 
                              double min_x = 0, double max_x = 4, 
                              double min_y = 0, double max_y = 4)
{
    int max_attempts = 10;
    while(max_attempts > 0)
    {
        Points points = generateRandomPoints(num_points, min_x, max_x, min_y, max_y);
        Polygon polygon = generatePolygon(points);
        if (bg::is_valid(polygon))  // Check if the polygon is valid
        {
            return polygon;
        }
        std::cout << "Polygon is invalid, retrying..." << std::endl;
        max_attempts--;
    }
    throw std::runtime_error("Failed to generate a valid polygon after maximum attempts.");
}

int main()
{
    Polygon polygon_1 = generateRandomPolygon();
    printPolygon(polygon_1, "1");
    Polygon polygon_2 = generateRandomPolygon();
    printPolygon(polygon_2, "2");
    // Check intersection
    if (checkPolygonIntersection(polygon_1, polygon_2))
    {
        // Print the intersection polygon points and area
        Polygon intersect_polygon = getOverlappingPolygon(polygon_1, polygon_2);
        printPolygon(intersect_polygon, "intersect");
        // Check if the origin point inside the polygon
        Point origin = {0, 0};
        if (bg::within(origin, polygon_1))
        {
            std::cout << "origin is inside the polygon" << std::endl;
        }
        else
        {
            std::cout << "origin is outside the polygon" << std::endl;
        }
    }
    else
    {
        std::cout << "Two polygons are not intersected" << std::endl;
    }
}
