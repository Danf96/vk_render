
#include <vk_types.h>
#include <SDL_events.h>

class Camera {
public:
	glm::vec3 velocity;
	glm::vec3 position;
	float pitch{ 0.f };
	float yaw{ 0.f };

	glm::mat4 getViewMatrix();
	glm::mat4 getRotationMatrix();

	void processSDLEvent(SDL_Event &e);
	void update();
};
