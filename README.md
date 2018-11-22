# AntNRE                                                                                                                                 
AntNRE is a neural entity relation extraction package built on PyTorch.
It aims to provide efficient and
flexible toolkits for building information extraction systems.

## Modularity
AntNRE contains modules which can be used as building blocks for various entity relation
extraction systems. For example,
- Encoders: CNN/RNN-based word representations, CNN/RNN-based sequence modelling.
- Entity Models: sequence labelling with RNN + CRF.
- Relation Models: Feature-enriched PCNN.

## Flexibility
One could use the package to implement many state-of-the-art entity relation extraction
systems. For example,
- Joint entity relation extraction model.
- Minimum risk training based model.
- Multi-task joint entity relation extraction model.

## Future Plans
More advanced systems will be integrated in the future. For example,
- Multi-instance multi-label models.
- Deep latent models.
