#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

class Solution 
{
public:
    int areaOfMaxDiagonal(const std::vector<std::vector<int>>& dimensions) 
    {
        std::unordered_map <float, float> diag_and_area;
        double max_diag = 0;
        for (size_t i = 0; i < dimensions.size(); i++)
        {
            double diag_length = diagnoalLengthSquare(dimensions[i]);
            if (diag_length >= max_diag)
            {
                double area = this->area(dimensions[i]);
                max_diag = diag_length;
                auto it = diag_and_area.find(max_diag);
                if (it != diag_and_area.end())
                {
                    if (diag_and_area[max_diag] > area)
                    {
                        area = diag_and_area[max_diag];
                    }
                    diag_and_area[max_diag] = area;
                }
                else
                {
                    diag_and_area.insert(std::make_pair(max_diag, area));
                }
            }
        }
        return diag_and_area[max_diag];
    }
    double diagnoalLengthSquare(const std::vector<int>& dimension)
    {
        if (dimension.size() == 2)
        {
            return pow(dimension[0], 2) + pow(dimension[1], 2);
        }
        else
        {
            throw std::runtime_error("input vector is invalid");
        }
    }
    double area(const std::vector<int>& dimension)
    {
        return dimension[0] * dimension[1];
    }
};

int main()
{
    std::vector<std::vector<int>> dimensions;
    dimensions.push_back(std::vector<int>{10, 3});
    dimensions.push_back(std::vector<int>{5, 9});
    dimensions.push_back(std::vector<int>{8, 3});
    Solution solution;
    int area = solution.areaOfMaxDiagonal(dimensions);
    std::cout << "result is: " << area << std::endl;
}