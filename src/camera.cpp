#include "camera.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

bool isFinite(const Vector3& value) {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

}

void Camera::updateCameraVectors() {
    const float radYaw = DirectX::XMConvertToRadians(yaw);
    const float radPitch = DirectX::XMConvertToRadians(pitch);
    front = Vector3(
        std::cos(radPitch) * std::cos(radYaw),
        std::sin(radPitch),
        std::cos(radPitch) * std::sin(radYaw));
    front.Normalize();

    right = front.Cross(worldUp);
    if (right.LengthSquared() <= 1e-6f) {
        right = Vector3(-std::sin(radYaw), 0.0f, std::cos(radYaw));
    } else {
        right.Normalize();
    }
    up = right.Cross(front);
    up.Normalize();
    isViewDirty = true;
}

void Camera::setPerspective(bool option) {
    const ProjectionType type = option ? PERSPECTIVE : ORTHOGRAPHIC;
    if (projectionType != type) {
        projectionType = type;
        isProjectionDirty = true;
    }
}

// Aspect = width / height
void Camera::setAspect(float aspect) {
    if (!std::isfinite(aspect) || aspect <= 0.0f) {
        throw std::invalid_argument("Camera aspect ratio must be finite and positive");
    }
    if (this->aspect != aspect) {
        this->aspect = aspect;
        width = aspect;
        isProjectionDirty = true;
    }
}

void Camera::setFovY(float fovY) {
    if (!std::isfinite(fovY) || fovY <= 0.0f || fovY >= 180.0f) {
        throw std::invalid_argument("Camera vertical field of view must be between 0 and 180 degrees");
    }
    if (this->fovY != fovY) {
        this->fovY = fovY;
        tanHalfFovy = std::tan(DirectX::XMConvertToRadians(fovY) * 0.5f);
        isProjectionDirty = true;
    }
}

void Camera::setNearClip(float nearClip) {
    if (!std::isfinite(nearClip) || nearClip <= 0.0f || nearClip >= f) {
        throw std::invalid_argument("Camera near clip must be positive and less than the far clip");
    }
    if (n != nearClip) {
        n = nearClip;
        isProjectionDirty = true;
    }
}

void Camera::setFarClip(float farClip) {
    if (!std::isfinite(farClip) || farClip <= n) {
        throw std::invalid_argument("Camera far clip must be greater than the near clip");
    }
    if (f != farClip) {
        f = farClip;
        isProjectionDirty = true;
    }
}

void Camera::setPosition(const Vector3& position) {
    if (!isFinite(position)) {
        throw std::invalid_argument("Camera position must be finite");
    }
    if (this->position != position) {
        this->position = position;
        target = position + front;
        isViewDirty = true;
    }
}

void Camera::setTarget(const Vector3& target) {
    if (!isFinite(target)) {
        throw std::invalid_argument("Camera target must be finite");
    }
    const Vector3 direction = target - position;
    const float directionLengthSquared = direction.LengthSquared();
    if (directionLengthSquared <= 1e-12f) {
        throw std::invalid_argument("Camera target must differ from its position");
    }
    if (this->target != target) {
        this->target = target;
        const Vector3 normalizedDirection =
            direction / std::sqrt(directionLengthSquared);
        yaw = DirectX::XMConvertToDegrees(std::atan2(normalizedDirection.z, normalizedDirection.x));
        pitch = DirectX::XMConvertToDegrees(std::asin(std::clamp(normalizedDirection.y, -1.0f, 1.0f)));
        updateCameraVectors();
    }
}

void Camera::lookAt(const Vector3& target) {
    setTarget(target);
}

Matrix Camera::getViewMatrix() {
    if (isViewDirty) {
        viewMatrix = Matrix::CreateLookAt(position, target, up);
        isViewDirty = false;
    }
    return viewMatrix;
}

Matrix Camera::getProjectionMatrix() {
    if (isProjectionDirty) {
        switch (projectionType) {
            case PERSPECTIVE:
                projectionMatrix = _getPerspectiveMatrix();
                break;
            case ORTHOGRAPHIC:
                projectionMatrix = _getOrthographicMatrix();
                break;
        }
        isProjectionDirty = false;
    }
    return projectionMatrix;
}

Matrix Camera::_getPerspectiveMatrix() {
    return Matrix::CreatePerspectiveFieldOfView(
        DirectX::XMConvertToRadians(fovY), aspect, n, f);
}

Matrix Camera::_getOrthographicMatrix() {
    const float halfWidth = width * 0.5f * zoomFactor;
    const float halfHeight = height * 0.5f * zoomFactor;
    return Matrix::CreateOrthographicOffCenter(
        -halfWidth, halfWidth, -halfHeight, halfHeight, n, f);
}

void Camera::reset() {
    position = Vector3(0.0f, 0.0f, 5.0f);
    yaw = -90.0f;
    pitch = 0.0f;
    fovY = 90.0f;
    tanHalfFovy = std::tan(DirectX::XMConvertToRadians(fovY) * 0.5f);
    aspect = 1.0f;
    n = 0.1f;
    f = 100.0f;
    width = 1.0f;
    height = 1.0f;
    zoomFactor = 5.0f;
    projectionType = PERSPECTIVE;

    updateCameraVectors();
    target = position + front;
    isProjectionDirty = true;
}

void Camera::processMouseMotion(float dx, float dy) {
    yaw += dx * mouseSensitivity;
    pitch -= dy * mouseSensitivity;
    pitch = std::clamp(pitch, -89.0f, 89.0f);
    updateCameraVectors();
    target = position + front;
}

void Camera::handleKeyPress(char wParam, float deltaTime) {
    const float amount = moveSpeed * deltaTime;
    if (wParam == 'w' || wParam == 'W') moveForward(amount);
    else if (wParam == 's' || wParam == 'S') moveForward(-amount);
    else if (wParam == 'a' || wParam == 'A') moveRight(-amount);
    else if (wParam == 'd' || wParam == 'D') moveRight(amount);
}

void Camera::moveForward(float amount) {
    const Vector3 offset = front * amount;
    position += offset;
    target += offset;
    isViewDirty = true;
}

void Camera::moveRight(float amount) {
    const Vector3 offset = right * amount;
    position += offset;
    target += offset;
    isViewDirty = true;
}

void Camera::handleScroll(float scrollY, float deltaTime) {
    if (projectionType == PERSPECTIVE) {
        setFovY(std::clamp(fovY + scrollY * 75.0f * deltaTime, 10.0f, 120.0f));
    } else {
        const float newZoom = zoomFactor * std::pow(0.25f, scrollY * deltaTime);
        zoomFactor = std::clamp(newZoom, 0.01f, 100.0f);
        isProjectionDirty = true;
    }
}

Ray Camera::generateRay(int x, int y, int width, int height) const {
    return generateRay(
        static_cast<float>(x) + 0.5f,
        static_cast<float>(y) + 0.5f,
        width,
        height);
}

Ray Camera::generateRay(float x, float y, int screenWidth, int screenHeight) const {
    if (screenWidth <= 0 || screenHeight <= 0) {
        throw std::invalid_argument("Ray generation requires positive screen dimensions");
    }

    const float ndcX = 2.0f * x / static_cast<float>(screenWidth) - 1.0f;
    const float ndcY = 1.0f - 2.0f * y / static_cast<float>(screenHeight);
    const float aspectRatio = static_cast<float>(screenWidth) / screenHeight;

    if (projectionType == PERSPECTIVE) {
        Vector3 direction = front
            + right * (ndcX * aspectRatio * tanHalfFovy)
            + up * (ndcY * tanHalfFovy);
        direction.Normalize();
        return {position, direction};
    }

    const float halfHeight = height * 0.5f * zoomFactor;
    const float halfWidth = width * 0.5f * zoomFactor;
    const Vector3 origin = position
        + right * (ndcX * halfWidth)
        + up * (ndcY * halfHeight);
    return {origin, front};
}

Camera::Camera() {
    reset();
}