#include <iostream>
#include <Eigen/Dense>
using namespace std;
using namespace Eigen;

Vector3d calcProjection(Vector3d A, Vector3d B, Vector3d P)
{
    Vector3d AB = B - A;
    Vector3d AP = P - A;
    float AC_norm = AB.dot(AP) / AB.norm();
    Vector3d C = A + AC_norm / AB.norm() * AB;
    return C;
}

// Suggestion: Pass by const reference avoid unnecessary copying
// const Vector3d& A
bool verifyProjection(Vector3d A, Vector3d B, Vector3d P, Vector3d C)
{
    Vector3d AB = B - A;
    Vector3d PC = P - C;
    Vector3d AC = C - A;
    Vector3d zero_vec = {0, 0, 0};
    Vector3d cross_product = AB.cross(AC);
    float dot_product = PC.dot(AB);
    // Suggestion: cross_product.isApprox(zero_vec)
    if (dot_product == 0 and cross_product == zero_vec)
    {
        return true;
    }
    else
    {
        return false;
    }
    // Suggestion return:
    // return (dot_product == 0 && cross_product.isApprox(zero_vec));
}

float calcDistance(const Vector3d A, const Vector3d B, const Vector3d P)
{
    Vector3d AB = B - A;
    Vector3d AP = P - A;
    Vector3d cross_product = AB.cross(AP);
    float area_parallelogram = cross_product.norm();
    return area_parallelogram / AB.norm();
}

int main()
{
    Eigen::Vector3d A = {0, 0, 0};
    Eigen::Vector3d B = {2, 0, 0};
    Eigen::Vector3d P = {1, 3, 0};

    Vector3d C = calcProjection(A, B, P);
    cout << "Projection point is: " << C.x() << ", " << C.y() << ", " << C.z() << endl;
    if (verifyProjection(A, B, P, C))
    {
        cout << "Verification passes!" << endl;
    }
    else
    {
        cout << "Verification failed." << endl;
    }

    cout << "Distance from P to AB is: " << calcDistance(A, B, P) << endl;

}

