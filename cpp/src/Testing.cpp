#include <iostream>
#include <string>
#include "NewComponent.h"
#include<opencv2/dnn/all_layers.hpp>
#include<opencv2/dnn.hpp>


using namespace std;

int main() {
    NewComponent component;
    //int sum = component.add(3, 5);
    //std::cout << "Sum: " << sum << std::endl;

    std::string greeting = component.greet();
    std::cout << greeting << std::endl;

    return 0;
}