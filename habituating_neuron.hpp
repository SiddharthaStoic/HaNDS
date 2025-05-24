#ifndef HABITUATING_NEURON_HPP
#define HABITUATING_NEURON_HPP

class HabituatingNeuron {
private:
    float response;
    float decay;
    float learning_rate;
    float threshold;

public:
    HabituatingNeuron(float decay = 0.05, float learning_rate = 0.2, float threshold = 0.3);

    // Update with binary stimulus (1 = present, 0 = absent)
    bool update(int stimulus);

    float get_response() const;
};

#endif