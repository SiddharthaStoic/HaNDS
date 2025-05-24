#include "habituating_neuron.hpp"
#include <algorithm>

HabituatingNeuron::HabituatingNeuron(float decay, float learning_rate, float threshold)
    : response(1.0), decay(decay), learning_rate(learning_rate), threshold(threshold) {}

bool HabituatingNeuron::update(int stimulus) {
    if (stimulus == 1) {
        response -= learning_rate * response;
    } else {
        response += decay * (1.0 - response);
    }

    // Clamp between 0 and 1
    response = std::max(0.0f, std::min(1.0f, response));
    return response < threshold;
}

float HabituatingNeuron::get_response() const {
    return response;
}