#include <iostream>
#include <vector>
#include <unordered_map>

using namespace std;

class Solution 
{
public:
    int maxFrequencyElements(vector<int>& nums) 
    {
        std::unordered_map<int, int> freq_counter;
        for(int num : nums)
        {
            freq_counter[num]++;
        }
        
        int max_frequency = 0;
        for (const auto& entry : freq_counter)
        {
            max_frequency = std::max(max_frequency, entry.second);
        }

        int max_freq_elements = 0;
        for (const auto& entry : freq_counter)
        {
            if (entry.second == max_frequency)
            {
                max_freq_elements++;
            }
        }

        int total_frequency = max_frequency * max_freq_elements;
        return total_frequency;
    }
};

int main()
{
    vector<int> num1 {7,7,7,1,2,2,3,4,4,1};
    Solution solution;
    int result = solution.maxFrequencyElements(num1);
    cout << "result is: " << result << endl;
}