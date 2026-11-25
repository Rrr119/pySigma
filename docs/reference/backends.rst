Backends
========

The backend module provides base classes for converting Sigma rules into target query
languages.

Backend
-------

.. autoclass:: sigma.conversion.base.Backend
   :members:
   :undoc-members:
   :show-inheritance:

TextQueryBackend
----------------

.. autoclass:: sigma.conversion.base.TextQueryBackend
   :members:
   :undoc-members:
   :show-inheritance:

ConversionState
---------------

.. autoclass:: sigma.conversion.state.ConversionState
   :members:
   :undoc-members:
   :show-inheritance:

DeferredQueryExpression
-----------------------

.. autoclass:: sigma.conversion.deferred.DeferredQueryExpression
   :members:
   :undoc-members:
   :show-inheritance:

Unified multi-backend conversion
--------------------------------

.. autoclass:: sigma.conversion.multi.UnifiedBackend
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: sigma.conversion.multi.BackendSpec
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: sigma.conversion.multi.convert_to_multiple_backends

.. autofunction:: sigma.conversion.multi.field_mapping_pipeline

Backend capabilities
--------------------

.. autoclass:: sigma.conversion.capabilities.BackendCapabilities
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: sigma.conversion.capabilities.Features
   :members:
   :undoc-members:

.. autofunction:: sigma.conversion.capabilities.detect_unsupported_features

Conversion reports
------------------

.. autoclass:: sigma.conversion.result.UnsupportedFeature
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: sigma.conversion.result.BackendConversionResult
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: sigma.conversion.result.MultiBackendResult
   :members:
   :undoc-members:
   :show-inheritance:

Semantic comparison
-------------------

.. autofunction:: sigma.conversion.semantic.semantic_model

.. autofunction:: sigma.conversion.semantic.semantic_models

.. autofunction:: sigma.conversion.semantic.semantically_equivalent
