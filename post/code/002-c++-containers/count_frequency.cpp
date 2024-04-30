#include <iostream>
#include <vector>

using namespace std;

class Solution 
{
public:
    int maxFrequencyElements(vector<int>& nums) 
    {
        vector<int> frequency (nums.size(), 0);
        for (int i = 0; i < frequency.size(); i ++)
        {
            frequency[i] = countDuplicatedNumber(i, nums);
        }
        for (int element : frequency)
        {
            std::cout << element << " ";
        }
        cout << endl;
        int max_value = checkMaxValue(frequency);
        return sumMaxValue(max_value, frequency);
    }
    int countDuplicatedNumber(const int& index, const vector<int>& vector)
    {
        int number = 1;
        for (int i = 1; i + index < vector.size(); i++)
        {
            if (vector[i + index] == vector[index])
            {
                number ++;
            }
        }
        return number;
    }
    int checkMaxValue(const vector<int>& vector)
    {
        int max = 0;
        for (int i = 0; i < vector.size(); i++)
        {
            if (vector[i] > max)
            {
                max = vector[i];
            }
        }
        cout << "max value in the vec is: " << max << endl;
        return max;
    }
    int sumMaxValue(const int& max, const vector<int>& vector)
    {
        int sum = 0;
        for (int i = 0; i < vector.size(); i++)
        {
            if (vector[i] == max)
            {
                sum += vector[i];
            }
        }
        return sum;
    }
};

int main()
{
    vector<int> num1 {1,2,2,3,4,4,1};
    Solution solution;
    float result = solution.maxFrequencyElements(num1);
    cout << "result is: " << result << endl;
}