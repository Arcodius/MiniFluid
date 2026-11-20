#pragma once

#include <SimpleMath.h>

#include <cstdint>

using Vector3 = DirectX::SimpleMath::Vector3;
using Matrix = DirectX::SimpleMath::Matrix;

struct Ray {
    Vector3 origin;
    Vector3 direction;
};

class Camera {
private:
    Vector3 position;
    Vector3 target;
    
    float yaw = -90.0f;   // 初始朝向为-z
    float pitch = 0.0f;   // 初始无俯仰
    Vector3 front = Vector3::Forward; // 相机朝向
    Vector3 right;
    Vector3 up; // 相机上方向
    Vector3 worldUp = Vector3::Up;

    float fovY;      // in degrees
    float tanHalfFovy; // 用于透视投影计算
    float aspect;    // width / height

    float n;
    float f;
    float width;
    float height;
    float zoomFactor = 5.f;

    Matrix viewMatrix;
    Matrix projectionMatrix;
    bool isViewDirty = true;
    bool isProjectionDirty = true;
    // Bumped on every change to the view or the projection. Unlike the dirty flags, which the matrix getters clear, a version can be watched by any number of readers: each one keeps the last version it saw and compares.
    uint64_t version = 0;

    void markViewDirty() { isViewDirty = true; ++version; }
    void markProjectionDirty() { isProjectionDirty = true; ++version; }

    float moveSpeed = 2.0f;
    float mouseSensitivity = 0.1f;

    Matrix _getPerspectiveMatrix();
    Matrix _getOrthographicMatrix();

    void updateCameraVectors();

    enum ProjectionType {
        PERSPECTIVE,
        ORTHOGRAPHIC
    };
    ProjectionType projectionType;
    
public:
    Camera();

    void reset();
    Vector3 getPosition() const { return position; }
    Vector3 getTarget() const { return target; }
    Vector3 getFront() const { return front; }
    Vector3 getRight() const { return right; }
    Vector3 getUp() const { return up; }
    float getFovY() const { return fovY; }
    float getAspect() const { return aspect; }
    float getOrthographicZoom() const { return zoomFactor; }
    bool isPerspective() const { return projectionType == PERSPECTIVE; }
    uint64_t getVersion() const { return version; }
    float getNearClip() const { return n; }
    float getFarClip() const { return f; }
    Matrix getViewMatrix();
    Matrix getProjectionMatrix();
    void setPosition(const Vector3& position);
    void setTarget(const Vector3& target);
    void lookAt(const Vector3& target);
	void setAspect(float aspect);
	void setFovY(float fovY);
	void setNearClip(float nearClip);
	void setFarClip(float farClip);
    void setPerspective(bool option);

    void processMouseMotion(float dx, float dy);
    void handleKeyPress(char wParam, float deltaTime);
    void moveForward(float amount);
    void moveRight(float amount);
    void handleScroll(float scrollY, float deltaTime);

    // Light tracing
    Ray generateRay(int x, int y, int width, int height) const;
    Ray generateRay(float x, float y, int screenWidth, int screenHeight) const;

};