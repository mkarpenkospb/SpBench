#include <env.hpp>
#include <utils.hpp>

namespace clbool {
    Controls create_controls(uint32_t platform_id, uint32_t device_id) {
        std::vector<cl::Platform> platforms;
        std::vector<cl::Device> devices;
        std::vector<cl::Kernel> kernels;
        cl::Program program;
        cl::Device device;
        try {
            cl::Platform::get(&platforms);
            platforms[platform_id].getDevices(CL_DEVICE_TYPE_GPU, &devices);
            return Controls(devices[device_id]);

        } catch (const cl::Error &e) {
            std::stringstream exception;
            exception << "\n" << e.what() << " : " << e.err() << "\n";
            throw std::runtime_error(exception.str());
        }
    }

    void show_devices() {
        std::vector<cl::Platform> platforms;
        std::vector<cl::Kernel> kernels;
        cl::Program program;
        cl::Device device;
        try {
            cl::Platform::get(&platforms);
            for (size_t i = 0; i < platforms.size(); ++i) {
                std::cout << "platform id: " << i << " \n";
                utils::printPlatformInfo(platforms[i]);
            }
        } catch (const cl::Error &e) {
            std::stringstream exception;
            exception << "\n" << e.what() << " : " << e.err() << "\n";
            throw std::runtime_error(exception.str());
        }
    }
}

