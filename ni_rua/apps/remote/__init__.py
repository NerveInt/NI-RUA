"""Remote GPU services reached through SSH local forwards (docs/development/so101-integration.md section 9).

`remote` in the NI-RUA config declares the host and the services (and how to start them); the gateway and
`python -m ni_rua.apps.remote --config <file>` connect, reuse or start each service, open the forwards and keep health
flowing (`manager.RemoteManager`). `python -m ni_rua.apps.remote --host H` keeps the original standalone forward without
starting anything (`tunnel.Supervisor`). Every run writes a provenance record (remote host, forwards, served versions,
started/reused). Nothing here enters model context. When a service is down the perception clients already degrade to
status unknown / unavailable.
"""
