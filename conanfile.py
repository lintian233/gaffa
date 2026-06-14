from conan import ConanFile
from conan.tools.cmake import CMakeDeps, CMakeToolchain


class GaffaConan(ConanFile):
    name = "gaffa"
    version = "0.1.0"
    package_type = "application"

    settings = "os", "arch", "compiler", "build_type"

    def requirements(self):
        self.requires("pybind11/2.13.6")
        self.requires("gtest/1.16.0")

    def build_requirements(self):
        self.tool_requires("cmake/3.31.6")

    def generate(self):
        toolchain = CMakeToolchain(self)
        toolchain.generate()

        deps = CMakeDeps(self)
        deps.generate()
