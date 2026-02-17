:html_theme.sidebar_secondary.remove:

.. title:: SPD Learn - Deep Learning on Riemannian Manifolds

.. raw:: html

    <style type="text/css">
    h1 {
      position: absolute !important;
      width: 1px !important;
      height: 1px !important;
      padding: 0 !important;
      margin: -1px !important;
      overflow: hidden !important;
      clip: rect(0, 0, 0, 0) !important;
      white-space: nowrap !important;
      border: 0 !important;
    }
    </style>

    <!-- SPD Cone Background Animation -->
    <div class="spd-background-container">
      <div class="spd-cone-bg spd-cone-bg-right">
        <img class="spd-cone-dark" src="_static/spd_cone_dark.gif" alt="" aria-hidden="true">
        <img class="spd-cone-light" src="_static/spd_cone_light.gif" alt="" aria-hidden="true">
      </div>
      <div class="spd-cone-bg spd-cone-bg-left">
        <img class="spd-cone-dark" src="_static/spd_cone_minimal.gif" alt="" aria-hidden="true">
        <img class="spd-cone-light" src="_static/spd_cone_minimal_light.gif" alt="" aria-hidden="true">
      </div>
    </div>

.. only:: html

   .. container:: hf-hero

      .. grid:: 1 1 2 2
         :gutter: 4
         :class-container: hf-hero-grid

         .. grid-item::
            :class: hf-hero-copy hf-reveal hf-delay-1

            .. rst-class:: hf-hero-title

               Deep Learning on Symmetric Positive Definite Matrices

            .. rst-class:: hf-hero-lede

               **SPD Learn** is a *pure* PyTorch library for geometric deep learning on Symmetric Positive Definite (SPD) matrices.
               The library provides differentiable Riemannian operations, broadcast-compatible layers,
               and reference implementations of published neural network architectures for SPD data.

            .. container:: hf-hero-actions

               .. button-ref:: installation
                  :ref-type: doc
                  :color: primary
                  :class: sd-btn-lg hf-btn hf-btn-primary

                  Get Started

               .. button-ref:: generated/auto_examples/index
                  :color: secondary
                  :class: sd-btn-lg hf-btn hf-btn-secondary

                  Examples

         .. grid-item::
            :class: hf-hero-panel hf-reveal hf-delay-2

            .. container:: hf-hero-card hf-quickstart

               .. rst-class:: hf-card-title

                  Quickstart

               .. tab-set::
                  :class: hf-code-tabs

                  .. tab-item:: Install

                     .. code-block:: bash

                        pip install spd_learn

                  .. tab-item:: Basic Usage

                     .. code-block:: python

                        import torch
                        from spd_learn.models import SPDNet

                        # Create SPDNet for 22-channel EEG, 4 classes
                        model = SPDNet(n_chans=22, n_outputs=4)

                        # Input: (batch, channels, time)
                        X = torch.randn(32, 22, 500)
                        output = model(X)  # (32, 4)

               .. rst-class:: hf-card-note

               Works with Python 3.11+ and PyTorch 2.0+

               .. container:: hf-card-actions

                  .. button-ref:: api
                     :color: secondary
                     :class: sd-btn-sm hf-btn hf-btn-ghost

                     API Reference

.. raw:: html

    <h2 class="hf-section-title hf-section-title-center">Why SPD Learn?</h2>
    <p class="hf-section-subtitle" style="text-align: center; margin-left: auto; margin-right: auto;">A PyTorch library providing differentiable Riemannian operations and neural network layers for SPD matrix-valued data.</p>

.. only:: html

   .. container:: hf-badges

      .. image:: https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue?style=flat-square
         :alt: Python Versions

      .. image:: https://img.shields.io/badge/license-BSD--3--Clause-green?style=flat-square
         :alt: License
         :target: https://github.com/spdlearn/spd_learn/blob/main/LICENSE.txt

      .. image:: https://img.shields.io/badge/PyTorch-2.0%2B-red?style=flat-square&logo=pytorch
         :alt: PyTorch

.. only:: html

   .. grid:: 1 2 3 3
      :gutter: 3
      :class-container: hf-stat-grid

      .. grid-item-card::
         :link: api
         :link-type: doc
         :text-align: left
         :class-card: hf-stat-card hf-reveal hf-delay-1

         .. raw:: html

            <div style="font-size: 2.5rem; margin-bottom: 0.5rem; color: var(--eegdash-primary);">
               <i class="fa-brands fa-python"></i>
            </div>

         **Pure PyTorch**

         .. rst-class:: hf-stat-text

            Built entirely on PyTorch, with automatic differentiation, and GPU acceleration.

      .. grid-item-card::
         :link: generated/auto_examples/index
         :link-type: doc
         :text-align: left
         :class-card: hf-stat-card hf-reveal hf-delay-2

         .. raw:: html

            <div style="font-size: 2.5rem; margin-bottom: 0.5rem; color: var(--eegdash-primary);">
               <i class="fa-solid fa-globe"></i>
            </div>

         **Riemannian Geometry**

         .. rst-class:: hf-stat-text

            Efficient exponential maps, logarithms, parallel transport, and geodesic distance computations on SPD manifolds.

      .. grid-item-card::
         :link: api
         :link-type: doc
         :text-align: left
         :class-card: hf-stat-card hf-reveal hf-delay-3

         .. raw:: html

            <div style="font-size: 2.5rem; margin-bottom: 0.5rem; color: var(--eegdash-primary);">
               <i class="fa-solid fa-cubes"></i>
            </div>

         **Model Zoo**

         .. rst-class:: hf-stat-text

            Implementations of SPDNet, TensorCSPNet, EEGSPDNet, TSMNet, and more state-of-the-art architectures.

----

.. raw:: html

    <h2 class="hf-section-title hf-section-title-center">Model Architectures</h2>
    <p class="hf-section-subtitle" style="text-align: center; margin-left: auto; margin-right: auto;">State-of-the-art deep learning models for SPD matrix data.</p>

.. only:: html

   .. grid:: 1 2 2 3
      :gutter: 4
      :class-container: hf-dataset-grid

      .. grid-item-card:: SPDNet
         :link: generated/spd_learn.models.SPDNet
         :link-type: doc
         :class-card: hf-dataset-card

         .. image:: _static/models/spdnet.png
            :alt: SPDNet Architecture
            :class: sd-card-img-top

         The cornerstone architecture for deep learning on non-Euclidean geometry of SPD manifolds. Performs dimension reduction while preserving the SPD structure.

         .. container:: hf-card-tags

            .. raw:: html

               <span class="hf-tag">BiMap</span>
               <span class="hf-tag">ReEig</span>
               <span class="hf-tag">LogEig</span>

      .. grid-item-card:: EEGSPDNet
         :link: generated/spd_learn.models.EEGSPDNet
         :link-type: doc
         :class-card: hf-dataset-card

         .. image:: _static/models/eegspdnet.jpeg
            :alt: EEGSPDNet Architecture
            :class: sd-card-img-top

         Specialized for EEG signal classification. Combines convolutional with covariance SPD network layers for Brain-Computer Interface applications.

         .. container:: hf-card-tags

            .. raw:: html

               <span class="hf-tag">Covariance</span>
               <span class="hf-tag">BiMap</span>
               <span class="hf-tag">ReEig</span>

      .. grid-item-card:: TSMNet
         :link: generated/spd_learn.models.TSMNet
         :link-type: doc
         :class-card: hf-dataset-card

         .. image:: _static/models/tsmnet.png
            :alt: TSMNet Architecture
            :class: sd-card-img-top

         Tangent Space Mapping Network (TSMNet) combining convolutional features with SPD batch normalization.

         .. container:: hf-card-tags

            .. raw:: html

               <span class="hf-tag">BatchNorm</span>
               <span class="hf-tag">LogEig</span>
               <span class="hf-tag">Transfer</span>

      .. grid-item-card:: TensorCSPNet
         :link: generated/spd_learn.models.TensorCSPNet
         :link-type: doc
         :class-card: hf-dataset-card

         .. image:: _static/models/tensorcspnet.png
            :alt: TensorCSPNet Architecture
            :class: sd-card-img-top

         Filterbank SPDNet with Tensor Common Spatial Patterns for multi-band EEG feature extraction.

         .. container:: hf-card-tags

            .. raw:: html

               <span class="hf-tag">Multi-band</span>
               <span class="hf-tag">CSP</span>
               <span class="hf-tag">BiMap</span>

      .. grid-item-card:: PhaseSPDNet
         :link: generated/spd_learn.models.PhaseSPDNet
         :link-type: doc
         :class-card: hf-dataset-card

         .. image:: _static/models/phase_spdnet.png
            :alt: PhaseSPDNet Architecture
            :class: sd-card-img-top

         Phase-based SPDNet that leverages phase information from the signals.

         .. container:: hf-card-tags

            .. raw:: html

               <span class="hf-tag">Phase</span>
               <span class="hf-tag">Dynamic System</span>
               <span class="hf-tag">BiMap</span>

      .. grid-item-card:: Green
         :link: generated/spd_learn.models.Green
         :link-type: doc
         :class-card: hf-dataset-card

         .. image:: _static/models/green.png
            :alt: Green Architecture
            :class: sd-card-img-top

         Gabor Riemann EEGNet combining Gabor wavelets with Riemannian geometry for interpretable EEG decoding.

         .. container:: hf-card-tags

            .. raw:: html

               <span class="hf-tag">Gabor</span>
               <span class="hf-tag">Wavelet</span>
               <span class="hf-tag">Shrinkage</span>

      .. grid-item-card:: MAtt
         :link: generated/spd_learn.models.MAtt
         :link-type: doc
         :class-card: hf-dataset-card

         .. image:: _static/models/matt.png
            :alt: MAtt Architecture
            :class: sd-card-img-top

         Manifold Attention Network that applies attention mechanisms directly on the SPD manifold using Log-Euclidean distances for temporal weighting.

         .. container:: hf-card-tags

            .. raw:: html

               <span class="hf-tag">Attention</span>
               <span class="hf-tag">LogEuclidean</span>
               <span class="hf-tag">TraceNorm</span>

----

.. raw:: html

    <h2 class="hf-section-title hf-section-title-center">Key Features</h2>
    <p class="hf-section-subtitle" style="text-align: center; margin-left: auto; margin-right: auto;">Core components for constructing and training geometric neural networks on SPD manifolds.</p>

.. only:: html

   .. grid:: 1 2 3 3
      :gutter: 4
      :class-container: hf-features-grid

      .. grid-item-card::
         :class-card: hf-feature-card

         .. raw:: html

            <div class="hf-feature-icon">
               <i class="fa-solid fa-layer-group"></i>
            </div>

         **SPD Layers**

         Specialized neural network layers for SPD matrices: BiMap for bilinear mappings, ReEig for eigenvalue rectification, and LogEig for tangent space projection.

      .. grid-item-card::
         :class-card: hf-feature-card

         .. raw:: html

            <div class="hf-feature-icon">
               <i class="fa-solid fa-globe"></i>
            </div>

         **Riemannian Operations**

         Complete toolkit for SPD manifold computations: exponential/logarithmic maps, geodesic distances, Log-Euclidean mean, and geodesic interpolation.

      .. grid-item-card::
         :class-card: hf-feature-card

         .. raw:: html

            <div class="hf-feature-icon">
               <i class="fa-solid fa-bolt"></i>
            </div>

         **GPU Accelerated**

         Full CUDA support with efficient batched operations. Leverage PyTorch's automatic differentiation for seamless gradient computation on manifolds.

      .. grid-item-card::
         :class-card: hf-feature-card

         .. raw:: html

            <div class="hf-feature-icon">
               <i class="fa-solid fa-diagram-project"></i>
            </div>

         **scikit-learn Compatible**

         Seamlessly integrate with scikit-learn pipelines, cross-validation, and hyperparameter tuning via skorch/Braindecode wrappers, with many tutorials provided.

      .. grid-item-card::
         :class-card: hf-feature-card

         .. raw:: html

            <div class="hf-feature-icon">
               <i class="fa-solid fa-chart-line"></i>
            </div>

         **Batch Normalization**

         SPD-specific batch normalization layers that respect the Riemannian geometry, enabling stable training of deep SPD networks.

      .. grid-item-card::
         :class-card: hf-feature-card

         .. raw:: html

            <div class="hf-feature-icon">
               <i class="fa-solid fa-code-branch"></i>
            </div>

         **Open Source**

         BSD-3-Clause licensed, for friendly for commercially applications, with actively maintained by experts from the field, and welcoming contributions.

----

.. raw:: html

    <h2 class="hf-section-title hf-section-title-center">Getting Started</h2>
    <p class="hf-section-subtitle" style="text-align: center; margin-left: auto; margin-right: auto;">Get up and running with SPD Learn in three steps.</p>

.. only:: html

   .. tab-set::
      :class: hf-getting-started-stepper

      .. tab-item:: 1. Install

         .. raw:: html

            <div class="step-header">
               <span class="step-indicator">Step 1 of 3</span>
               <h3 class="step-title">Install the library</h3>
               <p class="step-subtitle">Add SPD Learn to your Python environment with pip or from source.</p>
            </div>

         .. code-block:: bash

            pip install spd_learn

         .. raw:: html

            <p class="step-hint">For development, clone the repo and install in editable mode:</p>

         .. code-block:: bash

            git clone https://github.com/spdlearn/spd_learn
            cd spd_learn && pip install -e .

         .. raw:: html

            <div class="step-actions">
               <a href="installation.html" class="step-cta">Full installation guide</a>
            </div>

      .. tab-item:: 2. Import & Create

         .. raw:: html

            <div class="step-header">
               <span class="step-indicator">Step 2 of 3</span>
               <h3 class="step-title">Create your SPD model</h3>
               <p class="step-subtitle">Define an SPDNet with BiMap and ReEig layers in a few lines of code.</p>
            </div>

         .. code-block:: python

            from spd_learn.models import SPDNet

            model = SPDNet(
                n_chans=22,  # EEG channels
                n_outputs=4,  # Number of classes
                subspacedim=16,  # SPD subspace dimension
            )

         .. raw:: html

            <p class="step-hint">The model handles covariance computation, SPD projection, and classification end-to-end.</p>
            <div class="step-actions">
               <a href="api.html" class="step-cta">Explore model architectures</a>
            </div>

      .. tab-item:: 3. Train

         .. raw:: html

            <div class="step-header">
               <span class="step-indicator">Step 3 of 3</span>
               <h3 class="step-title">Train with PyTorch</h3>
               <p class="step-subtitle">Use standard PyTorch training loops — no special APIs required.</p>
            </div>

         .. code-block:: python

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = torch.nn.CrossEntropyLoss()

            for epoch in range(100):
                optimizer.zero_grad()
                loss = criterion(model(X_train), y_train)
                loss.backward()
                optimizer.step()

         .. raw:: html

            <p class="step-hint">SPD Learn integrates with scikit-learn pipelines via skorch wrappers.</p>
            <div class="step-actions">
               <a href="generated/auto_examples/index.html" class="step-cta">See full examples</a>
            </div>

----

.. raw:: html

    <h2 class="hf-section-title hf-section-title-center">Ecosystem Integration</h2>
    <p class="hf-section-subtitle" style="text-align: center; margin-left: auto; margin-right: auto;">Works seamlessly with your favorite tools.</p>

.. only:: html

   .. grid:: 2 3 4 4
      :gutter: 3
      :class-container: hf-ecosystem-grid

      .. grid-item::
         :class: hf-ecosystem-item

         .. raw:: html

            <div class="hf-ecosystem-card">
               <i class="fa-brands fa-python" style="font-size: 3rem; color: #3776AB;"></i>
               <p class="hf-ecosystem-name">Python 3.11+</p>
            </div>

      .. grid-item::
         :class: hf-ecosystem-item

         .. raw:: html

            <div class="hf-ecosystem-card">
               <img src="_static/pytorch-logo.png" alt="PyTorch" style="height: 52px;">
               <p class="hf-ecosystem-name">PyTorch</p>
            </div>

      .. grid-item::
         :class: hf-ecosystem-item

         .. raw:: html

            <div class="hf-ecosystem-card">
               <img src="_static/sklearn-logo.svg" alt="scikit-learn" style="height: 52px;">
               <p class="hf-ecosystem-name">scikit-learn</p>
            </div>

      .. grid-item::
         :class: hf-ecosystem-item

         .. raw:: html

            <div class="hf-ecosystem-card">
               <i class="fa-solid fa-brain" style="font-size: 3rem; color: #8B5CF6;"></i>
               <p class="hf-ecosystem-name">pyRiemann</p>
            </div>

----

.. only:: html

   .. container:: hf-callout hf-reveal hf-delay-2

      .. rst-class:: hf-callout-title

         Open Source & Community Driven

      .. rst-class:: hf-callout-text

         SPD Learn is an open-source project contributed by researchers for researchers.
         Join our community and help advance deep learning on Riemannian manifolds.

      .. container:: hf-callout-actions

         .. button-link:: https://github.com/spdlearn/spd_learn
            :color: secondary
            :class: sd-btn-lg hf-btn hf-btn-secondary

            GitHub

         .. button-link:: https://github.com/spdlearn/spd_learn/issues
            :color: secondary
            :class: sd-btn-lg hf-btn hf-btn-secondary

            Report Issues

      .. rst-class:: hf-callout-support

         Supported by

      .. raw:: html

         <div class="logos-container hf-logo-cloud">
           <div class="logo-item">
             <img src="_static/support/inria_red.png" alt="Inria" class="logo-light" style="height: 300px; width: auto;">
             <img src="_static/support/inria_red.png" alt="Inria" class="logo-dark" style="height: 300px; width: auto;">
           </div>
           <div class="logo-item">
             <img src="_static/support/cnrs_dark.png" alt="CNRS" class="logo-light" style="height:  300px; width: auto;">
             <img src="_static/support/cnrs_white.png" alt="CNRS" class="logo-dark" style="height:  300px; width: auto;">
           </div>
           <div class="logo-item">
             <img src="_static/support/cea_dark.png" alt="CEA" class="logo-light" style="height:  300px; width: auto;">
             <img src="_static/support/cea_white.png" alt="CEA" class="logo-dark" style="height:  300px; width: auto;">
           </div>
           <div class="logo-item">
             <img src="_static/support/paris_saclay_dark.png" alt="Université Paris-Saclay" class="logo-light" style="height: 300px; width: auto;">
             <img src="_static/support/paris_saclay_white.png" alt="Université Paris-Saclay" class="logo-dark" style="height: 300px; width: auto;">
           </div>
           <div class="logo-item">
             <img src="_static/support/atr_logo_white.png" alt="ATR" class="logo-light" style="height: 300px; width: auto;">
             <img src="_static/support/atr_logo_dark.png" alt="ATR" class="logo-dark" style="height: 300px; width: auto;">
           </div>
           <div class="logo-item">
             <img src="_static/support/ucsd_dark.svg" alt="UC San Diego" class="logo-light" style="height: 300px; width: auto;">
             <img src="_static/support/ucsd_dark.svg" alt="UC San Diego" class="logo-dark" style="height: 300px; width: auto;">
           </div>
           <div class="logo-item">
             <img src="_static/support/usmb_dark.png" alt="Université Savoie Mont Blanc" class="logo-light" style="height: 300px; width: auto;">
             <img src="_static/support/usmb_dark.png" alt="Université Savoie Mont Blanc" class="logo-dark" style="height: 300px; width: auto;">
           </div>
         </div>

----

.. raw:: html

    <h2 class="hf-section-title hf-section-title-center">Citation</h2>
    <p class="hf-section-subtitle" style="text-align: center; margin-left: auto; margin-right: auto;">If you use SPD Learn in your research, please cite us.</p>

.. raw:: html

    <div class="citation-apa">
      <span class="citation-text">Aristimunha, B., Ju, C., Collas, A., Bouchard, F., Mian, A., Thirion, B., Chevallier, S., &amp; Kobler, R. (2026). SPDlearn: A geometric deep learning Python library for neural decoding through trivialization. <em>To be submitted</em>. https://github.com/spdlearn/spd_learn</span>
      <button class="citation-copy-btn" onclick="navigator.clipboard.writeText('Aristimunha, B., Ju, C., Collas, A., Bouchard, F., Mian, A., Thirion, B., Chevallier, S., & Kobler, R. (2026). SPDlearn: A geometric deep learning Python library for neural decoding through trivialization. To be submitted. https://github.com/spdlearn/spd_learn'); this.innerHTML='<i class=&quot;fa-solid fa-check&quot;></i>'; setTimeout(() => this.innerHTML='<i class=&quot;fa-solid fa-copy&quot;></i>', 2000);" title="Copy APA citation">
        <i class="fa-solid fa-copy"></i>
      </button>
    </div>

.. code-block:: bibtex

   @article{aristimunha2025spdlearn,
     title = {SPDlearn: A Geometric Deep Learning Python Library for Neural Decoding Through Trivialization},
     author = {Aristimunha, Bruno and Ju, Ce and Collas, Antoine and Bouchard, Florent and Mian, Ammar and Thirion, Bertrand and Chevallier, Sylvain and Kobler, Reinmar},
     journal = {to be submitted},
     year = {2026},
     url = {https://github.com/spdlearn/spd_learn}
   }

.. toctree::
   :hidden:

   Installation <installation>
   User Guide <user_guide>
   Theory <theory>
   API <api>
   Examples <generated/auto_examples/index>
   FAQ <faq>
   Contributing <contributing>
