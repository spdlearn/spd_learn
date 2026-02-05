:html_theme.sidebar_secondary.remove: true

.. _theory:

======
Theory
======

.. rst-class:: theory-intro

   Mathematical foundations and reference material for SPD Learn.

.. only:: html

   .. raw:: html

      <div class="theory-grid">
         <a href="background/index.html" class="theory-card">
            <div class="theory-card-header">
               <div class="theory-card-icon">
                  <i class="fa-solid fa-book-open"></i>
               </div>
               <span class="theory-card-tag tag-start">Start here</span>
            </div>
            <h3 class="theory-card-title">Background</h3>
            <ul class="theory-card-list">
               <li>Data representations (EEG/fMRI covariances)</li>
               <li>Minimal geometry & metric choices</li>
               <li>SPDNet pipeline overview</li>
            </ul>
            <div class="theory-card-footer">
               <span class="theory-card-meta">~10 min</span>
               <span class="theory-card-cta">Read <i class="fa-solid fa-arrow-right"></i></span>
            </div>
         </a>

         <a href="geometric_concepts.html" class="theory-card">
            <div class="theory-card-header">
               <div class="theory-card-icon">
                  <i class="fa-solid fa-compass-drafting"></i>
               </div>
               <span class="theory-card-tag tag-core">Core math</span>
            </div>
            <h3 class="theory-card-title">Geometric Concepts</h3>
            <ul class="theory-card-list">
               <li>SPD manifold & Riemannian metrics</li>
               <li>Exp/Log maps, parallel transport</li>
               <li>Layer operation visualizations</li>
            </ul>
            <div class="theory-card-footer">
               <span class="theory-card-meta">~20 min</span>
               <span class="theory-card-cta">Read <i class="fa-solid fa-arrow-right"></i></span>
            </div>
         </a>

         <a href="numerical_stability.html" class="theory-card">
            <div class="theory-card-header">
               <div class="theory-card-icon">
                  <i class="fa-solid fa-shield-halved"></i>
               </div>
               <span class="theory-card-tag tag-practical">Practical</span>
            </div>
            <h3 class="theory-card-title">Numerical Stability</h3>
            <ul class="theory-card-list">
               <li>Dtype-aware thresholds & clamping</li>
               <li>Configuration for stability</li>
               <li>Troubleshooting NaN & convergence</li>
            </ul>
            <div class="theory-card-footer">
               <span class="theory-card-meta">~8 min</span>
               <span class="theory-card-cta">Read <i class="fa-solid fa-arrow-right"></i></span>
            </div>
         </a>

         <a href="notation.html" class="theory-card">
            <div class="theory-card-header">
               <div class="theory-card-icon">
                  <i class="fa-solid fa-sigma"></i>
               </div>
               <span class="theory-card-tag tag-reference">Reference</span>
            </div>
            <h3 class="theory-card-title">Notation</h3>
            <ul class="theory-card-list">
               <li>Manifold symbols (𝒮⁺, Sym, T_P)</li>
               <li>Distance & metric conventions</li>
               <li>Layer operation symbols</li>
            </ul>
            <div class="theory-card-footer">
               <span class="theory-card-meta">~3 min</span>
               <span class="theory-card-cta">Read <i class="fa-solid fa-arrow-right"></i></span>
            </div>
         </a>

         <a href="glossary.html" class="theory-card">
            <div class="theory-card-header">
               <div class="theory-card-icon">
                  <i class="fa-solid fa-tags"></i>
               </div>
               <span class="theory-card-tag tag-reference">Reference</span>
            </div>
            <h3 class="theory-card-title">Glossary</h3>
            <ul class="theory-card-list">
               <li>SPD matrices & Riemannian terms</li>
               <li>Layer & module definitions</li>
               <li>Model architecture names</li>
            </ul>
            <div class="theory-card-footer">
               <span class="theory-card-meta">Quick lookup</span>
               <span class="theory-card-cta">Read <i class="fa-solid fa-arrow-right"></i></span>
            </div>
         </a>

         <a href="references.html" class="theory-card">
            <div class="theory-card-header">
               <div class="theory-card-icon">
                  <i class="fa-solid fa-bookmark"></i>
               </div>
               <span class="theory-card-tag tag-reference">Reference</span>
            </div>
            <h3 class="theory-card-title">References</h3>
            <ul class="theory-card-list">
               <li>Interactive literature map</li>
               <li>Model family legend</li>
               <li>Full bibliography & BibTeX</li>
            </ul>
            <div class="theory-card-footer">
               <span class="theory-card-meta">95 papers</span>
               <span class="theory-card-cta">Read <i class="fa-solid fa-arrow-right"></i></span>
            </div>
         </a>
      </div>


.. toctree::
   :maxdepth: 2
   :hidden:

   background/index
   geometric_concepts
   numerical_stability
   notation
   glossary
   references


.. seealso::

   - :doc:`user_guide` -- Getting started with SPD Learn
   - :doc:`api` -- Complete API reference
   - :doc:`generated/auto_examples/index` -- Practical examples
