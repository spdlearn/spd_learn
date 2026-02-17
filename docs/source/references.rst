:html_theme.sidebar_secondary.remove: true

.. _references:

==========
References
==========

This page provides an overview of the foundational literature behind SPD Learn,
organized by model architecture and research lineage.

Literature Map
==============

The following visualization shows the relationships between key publications
implemented in SPD Learn. The map illustrates how different SPD neural network
architectures have evolved and influenced each other over time.

.. figure:: _static/litmap_spd_learn.png
   :alt: Literature map showing connections between SPD Learn model papers
   :align: center
   :width: 100%
   :class: litmap-figure

   **SPD Learn Literature Map** — Generated with `Litmaps <https://www.litmaps.com/>`_.
   The x-axis represents publication date (more recent to the right), and the
   y-axis represents citation count (more cited at the top). Lines show citation
   relationships between papers.

----

Model Legend
============

The following legend explains the color coding used in the literature map above,
with each color representing a distinct model family implemented in SPD Learn.

.. raw:: html

   <style>
   .legend-grid {
     display: grid;
     grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
     gap: 1rem;
     margin: 1.5rem 0;
   }
   .legend-item {
     display: flex;
     align-items: flex-start;
     gap: 0.75rem;
     padding: 1rem;
     border-radius: 12px;
     background: var(--pst-color-surface);
     border: 1px solid var(--pst-color-border);
     transition: transform 0.2s ease, box-shadow 0.2s ease;
   }
   .legend-item:hover {
     transform: translateY(-2px);
     box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
   }
   .legend-dot {
     width: 16px;
     height: 16px;
     border-radius: 50%;
     flex-shrink: 0;
     margin-top: 3px;
   }
   .legend-content {
     flex: 1;
   }
   .legend-title {
     font-weight: 600;
     font-size: 0.95rem;
     margin-bottom: 0.25rem;
   }
   .legend-desc {
     font-size: 0.85rem;
     color: var(--pst-color-text-muted);
     line-height: 1.4;
   }
   .legend-ref {
     font-size: 0.8rem;
     color: var(--pst-color-primary);
     margin-top: 0.25rem;
   }
   </style>

   <div class="legend-grid">
     <div class="legend-item">
       <div class="legend-dot" style="background: #7DD3FC;"></div>
       <div class="legend-content">
         <div class="legend-title">SPDNet</div>
         <div class="legend-desc">The foundational architecture for deep learning on SPD manifolds using BiMap, ReEig, and LogEig layers. Originally developed for computer vision tasks; application to EEG/neuroimaging is original work by the BCI research community.</div>
         <div class="legend-ref">Huang & Van Gool, 2017</div>
       </div>
     </div>
     <div class="legend-item">
       <div class="legend-dot" style="background: #047857;"></div>
       <div class="legend-content">
         <div class="legend-title">Riemannian BatchNorm</div>
         <div class="legend-desc">Riemannian batch normalization technique for SPD neural networks, used in TSMNet and TensorCSPNet.</div>
         <div class="legend-ref">Brooks et al., 2019</div>
       </div>
     </div>
     <div class="legend-item">
       <div class="legend-dot" style="background: #F97316;"></div>
       <div class="legend-content">
         <div class="legend-title">TSMNet</div>
         <div class="legend-desc">Tangent Space Mapping Network combining temporal convolutions with SPD batch normalization for domain adaptation.</div>
         <div class="legend-ref">Kobler et al., 2022</div>
       </div>
     </div>
     <div class="legend-item">
       <div class="legend-dot" style="background: #BEF264;"></div>
       <div class="legend-content">
         <div class="legend-title">MAtt</div>
         <div class="legend-desc">Manifold Attention Network integrating Riemannian geometry with attention mechanisms on SPD matrices.</div>
         <div class="legend-ref">Pan et al., 2022</div>
       </div>
     </div>
     <div class="legend-item">
       <div class="legend-dot" style="background: #A855F7;"></div>
       <div class="legend-content">
         <div class="legend-title">TensorCSPNet</div>
         <div class="legend-desc">Multi-band SPDNet framework using tensor stacking and Common Spatial Patterns for filter bank EEG classification.</div>
         <div class="legend-ref">Ju et al., 2022</div>
       </div>
     </div>
     <div class="legend-item">
       <div class="legend-dot" style="background: #FBBF24;"></div>
       <div class="legend-content">
         <div class="legend-title">EEGSPDNet</div>
         <div class="legend-desc">End-to-end architecture with channel-specific convolution, covariance pooling, and SPDNet for EEG classification.</div>
         <div class="legend-ref">Wilson et al., 2025</div>
       </div>
     </div>
     <div class="legend-item">
       <div class="legend-dot" style="background: #22C55E;"></div>
       <div class="legend-content">
         <div class="legend-title">GREEN</div>
         <div class="legend-desc">Gabor Riemann EEGNet combining learnable Gabor wavelets with Riemannian geometry and shrinkage estimation.</div>
         <div class="legend-ref">Paillard et al., 2025</div>
       </div>
     </div>
     <div class="legend-item">
       <div class="legend-dot" style="background: #3B82F6;"></div>
       <div class="legend-content">
         <div class="legend-title">PhaseSPDNet</div>
         <div class="legend-desc">Phase-Space Embedding combined with SPDNet for geometric analysis of EEG dynamics in reconstructed phase space.</div>
         <div class="legend-ref">Carrara et al., 2024</div>
       </div>
     </div>
   </div>

----

Full Bibliography
=================

Below are the complete bibliographic references for the models and foundational
works implemented in SPD Learn.

.. bibliography::
   :all:

----

BibTeX Entries
==============

For convenience, here are BibTeX entries for citing the main works:

.. code-block:: bibtex

   @inproceedings{huang2017riemannian,
     title={A Riemannian network for SPD matrix learning},
     author={Huang, Zhiwu and Van Gool, Luc},
     booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
     volume={31},
     number={1},
     year={2017}
   }

   @inproceedings{brooks2019riemannian,
     title={Riemannian batch normalization for SPD neural networks},
     author={Brooks, Daniel and Schwander, Olivier and Barbaresco, Fr{\'e}d{\'e}ric and
             Schneider, Jean-Yves and Cord, Matthieu},
     booktitle={Advances in Neural Information Processing Systems},
     volume={32},
     year={2019}
   }

   @inproceedings{kobler2022spd,
     title={SPD domain-specific batch normalization to crack interpretable
            unsupervised domain adaptation in EEG},
     author={Kobler, Reinmar J and Hirayama, Jun-ichiro and Zhao, Qibin and Kawanabe, Motoaki},
     booktitle={Advances in Neural Information Processing Systems},
     volume={35},
     pages={6219--6235},
     year={2022}
   }

   @inproceedings{pan2022matt,
     title={MAtt: A manifold attention network for EEG decoding},
     author={Pan, Yue-Ting and Chou, Jing-Lun and Wei, Chun-Shu},
     booktitle={Advances in Neural Information Processing Systems},
     volume={35},
     pages={31116--31129},
     year={2022}
   }

   @article{paillard2024green,
     title={GREEN: a lightweight architecture using learnable wavelets and
            Riemannian geometry for biomarker exploration},
     author={Paillard, Joseph and Hipp, Joerg F. and Engemann, Denis A.},
     journal={Patterns},
     volume={6},
     number={1},
     pages={101153},
     year={2025},
     doi={10.1016/j.patter.2025.101153}
   }

   @article{ju2022tensor,
     title={Tensor-CSPNet: A Novel Geometric Deep Learning Framework for
            Motor Imagery Classification},
     author={Ju, Ce and Guan, Cuntai},
     journal={IEEE Transactions on Neural Networks and Learning Systems},
     volume={34},
     number={12},
     pages={10955--10969},
     year={2023},
     doi={10.1109/TNNLS.2022.3172108}
   }

   @article{carrara2024eegspd,
     title={Geometric neural network based on phase space for BCI-EEG decoding},
     author={Carrara*, Igor and Aristimunha*, Bruno and Corsi, Marie-Constance and
             de Camargo, Raphael Y. and Chevallier, Sylvain and Papadopoulo, Th{\'e}odore},
     journal={Journal of Neural Engineering},
     volume={21},
     number={6},
     pages={016049},
     year={2024},
     doi={10.1088/1741-2552/ad88a2}
   }

  @article{ju2026spdmatrixlearningneuroimaging,
        title={SPD Matrix Learning for Neuroimaging Analysis: Perspectives, Methods, and Challenges},
        author={Ce Ju and Reinmar Kobler and Antoine Collas and Motoaki Kawanabe and Cuntai Guan and Bertrand Thirion},
        year={2026},
        journal={arXiv preprint arXiv2504.18882},
        url={https://arxiv.org/abs/2504.18882},
  }



----

Citing SPD Learn
================

If you use SPD Learn in your research, please cite:

.. code-block:: bibtex

   @article{aristimunha2025spdlearn,
     title={SPDlearn: A Geometric Deep Learning Python Library for
            Neural Decoding Through Trivialization},
     author={Aristimunha, Bruno and Ju, Ce and Collas, Antoine and
             Bouchard, Florent and Thirion, Bertrand and
             Chevallier, Sylvain and Kobler, Reinmar},
     journal={To be submitted},
     year={2026},
     url={https://github.com/spdlearn/spd_learn}
   }


.. seealso::

   - :doc:`theory` -- Theoretical foundations
   - :doc:`api` -- API reference for implemented models
   - :doc:`generated/auto_examples/index` -- Practical examples
