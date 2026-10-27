# Deploying the farm's cloud database (Amazon DynamoDB)

The application connects every pod's **identity** (the number printed on its tag) to its **crop type, growth stage,
inspections, disease status, treatment history and harvest readiness**, in one DynamoDB table. **No images are stored
there** - photos stay on the computer; the cloud gets numbers and text only (see "What is and is not stored").

> **Honest status.** This was built and tested against `moto` (a mock of DynamoDB) - never against a real AWS account,
> which the build environment could not reach. The CloudFormation template passes `cfn-lint`, and the code paths are the
> same ones boto3 uses against AWS, but your first real run is the real test. Step 6 below checks it in two commands.

Without any of this the application still works: it falls back to a **local file** that behaves identically
(`dynamo_table_cloud.json`) and says so in the interface ("☁ local", not "☁ AWS").

## 1. Install

```bash
pip install -r requirements.txt          # includes boto3
```

## 2. AWS credentials (never put keys in code or in the repository)

Use any normal AWS method; the application never stores credentials, boto3 finds them itself.

```bash
aws configure                            # or: export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...
```

## 3. Create the table - once

**Option A (recommended): CloudFormation** - creates the table *and* a least-privilege policy.

```bash
aws cloudformation deploy \
  --template-file aws/farm_table.yaml \
  --stack-name farm-agent-data \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides TableName=FarmPods \
  --region us-east-1
aws cloudformation describe-stacks --stack-name farm-agent-data --query "Stacks[0].Outputs" --region us-east-1
```

**Option B: one command** (table only, no policy): `FARM_DDB_TABLE=FarmPods FARM_AWS_REGION=us-east-1 python cloud_cli.py create-table`

The table uses **on-demand billing** (nothing to size, an idle farm costs essentially nothing), has **point-in-time
recovery** on, and is **retained** if you delete the stack. Check current AWS pricing for your region; the whole 72-pod
demo farm is a few hundred KB of data.

## 4. Permissions

The application needs exactly five calls on the table and its indexes - nothing else (no scan, no delete, no table
admin). Attach the managed policy the stack created (its ARN is the `AppPolicyArn` output) to the user or role the
application runs as:

```bash
aws iam attach-user-policy --user-name YOUR_USER --policy-arn <AppPolicyArn>
```

Not using CloudFormation? `aws/iam_policy.json` is the same policy: replace `REGION` and `ACCOUNT_ID`.

## 5. Tell the application

```bash
export FARM_DDB_TABLE=FarmPods           # setting this is what switches the application to DynamoDB
export FARM_AWS_REGION=us-east-1
python app_ui.py
```

PowerShell: `$env:FARM_DDB_TABLE="FarmPods"; $env:FARM_AWS_REGION="us-east-1"`

Other settings: `FARM_CLOUD=local|aws|off` (force a backend), `FARM_DDB_ENDPOINT` (e.g. DynamoDB Local).

## 6. Check that it really works

```bash
python cloud_cli.py ping                 # OK  DynamoDB table 'FarmPods' (us-east-1): ACTIVE
python app_ui.py                         # start a run; the top bar shows  ☁ AWS ✓ <events sent>
python cloud_cli.py summary              # how many pods are in each state
python cloud_cli.py show R1-L-F1         # one pod: identity, crop, growth, disease, treatments, harvest, tag
python cloud_cli.py pods --status harvest_ready
```

In the application, **☁** (top bar) opens the Cloud database window: it reads the records back *from the table*.

## What is and is not stored

| Stored (per pod, under its tag number) | Never stored |
|---|---|
| identity: tag number, crop type, rail / station / side / slot | any image, frame, thumbnail or tag-reading picture |
| the tag the robot camera read: confirmed or not, confidence, views agreeing | the local path of a photo |
| growth stage + confidence, crop age, harvest clock | credentials |
| every inspection: stage, disease flag/name/confidence, the decision, the reason | |
| disease status: healthy, flagged, treating, verifying, cleared | |
| every robot work order (treatment, harvest, replant, verify...) and its outcome | |
| harvest readiness: not ready, ready, ordered, harvesting, harvested | |
| a **file name and SHA-256** of each inspected photo, so a cloud record can be matched to the photo on this computer | |

"No images" is enforced in code (`cloud_model.assert_no_images`): bytes, arrays, image-like field names, base64 and
oversized text are refused before any write - by the queue, and again by each backend.

## Behaviour you should know about

* **The farm never waits on the network.** Writes happen in a background thread. If AWS is unreachable the events are
  kept in `dynamo_table_cloud_outbox.jsonl` and sent, in order and without double counting, when it is back (the
  interface shows `☁ OFFLINE · n waiting`; `python cloud_cli.py sync` forces a retry).
* **The application never deletes cloud data.** "Reset history" only clears the local history file. The cloud keeps its
  own cumulative record (so its visit counts will exceed the local ones after a reset). Delete the table by hand to start over.
* **One writer.** The pod profile is maintained by a single application instance. Two applications writing the same pods
  at the same time could miscount; one farm controller is the intended design.
* **Keys.** `pk = POD#<tag number>`; `sk = PROFILE | INSPECTION#<time>#<visit> | ORDER#<created>#<id> | TAGREAD#<time>`.
  Two indexes: `gsi_crop` ("every lettuce pod") and `gsi_status` ("every harvest_ready / disease_flagged /
  identity_unconfirmed pod"). Pod `status` values, most urgent first: `identity_unconfirmed`, `treatment_in_progress`,
  `disease_flagged`, `verifying_treatment`, `harvest_in_progress`, `harvest_ready`, `harvested`, `growing`.

## Troubleshooting

| Message | Meaning / fix |
|---|---|
| `ResourceNotFoundException` | the table does not exist in that region: step 3, or fix `FARM_AWS_REGION` |
| `AccessDeniedException` | the credentials lack the policy: step 4 |
| `NoCredentialsError` / `NoRegionError` | steps 2 and 5 |
| `☁ OFFLINE` in the top bar | nothing is lost; fix the cause above, then `python cloud_cli.py sync` |

---

# Part 2: the farm API (read API + observation/work-order Lambda)

A public URL that serves the records the application wrote, for a browser, a script, or a competition judge:

```
GET /pods/R1-L-F1                  one pod: profile + inspections, treatments, harvests, tag reads
GET /pods/R1-L-F1?history=0        just the profile        (?limit=1..50 sets how many history items)
GET /pods?status=harvest_ready     which pods need what     (statuses: see Part 1)
GET /pods?crop=lettuce             every lettuce pod
GET /pods                          every pod, compact, with counts per status
GET /health
POST /observations                 submit OpenCV metadata; Lambda stores it and returns a work order
```

`API Gateway (HTTP API) -> read Lambda + observation Lambda -> DynamoDB`. The read Lambda has only `GetItem`/`Query`; the
separate observation Lambda validates OpenCV metadata, stores observation/order/profile records, and selects `TREAT`,
`HARVEST`, `VERIFY`, or `MONITOR`. It refuses image-like payloads: photos stay on the local machine. The API is throttled
(10 requests/second by default). The write endpoint is disabled unless you configure its separate secret.

## Deploy (after Part 1, in the same account and region)

No zip, no S3 bucket, no SAM: the Lambda's code is written into the template.

```bash
export OBSERVATION_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
aws cloudformation deploy \
  --template-file aws/farm_api.yaml \
  --stack-name farm-agent-api \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides TableName=FarmPods ObservationApiKey="$OBSERVATION_API_KEY" \
  --region us-east-1
aws cloudformation describe-stacks --stack-name farm-agent-api --region us-east-1 --query "Stacks[0].Outputs"
```

The outputs give `ApiUrl` (the base URL), `TryIt`, and `ObservationEndpoint`. The GET API can remain public while writes
require the secret. Do not commit the secret. Set the local app's connection like this:

```bash
export FARM_LAMBDA_API_URL="$(aws cloudformation describe-stacks --stack-name farm-agent-api --region us-east-1 --query "Stacks[0].Outputs[?OutputKey=='ApiUrl'].OutputValue" --output text)"
export FARM_LAMBDA_API_KEY="$OBSERVATION_API_KEY"
```

Then:

```bash
API=https://abc123.execute-api.us-east-1.amazonaws.com        # your ApiUrl
curl "$API/health"
curl "$API/pods/R1-L-F1"
curl "$API/pods?status=harvest_ready"
curl -X POST "$API/observations" \
  -H "content-type: application/json" -H "x-api-key: $OBSERVATION_API_KEY" \
  -d '{"observation_id":"manual-test-12345678","pod_id":"R1-L-F1","crop_type":"lettuce","timestamp":"2026-10-26T10:00:00+00:00","visit":1,"growth_stage":"stage 04- Harvest stage","growth_confidence":0.9,"disease_flag":false,"disease_confidence":0.1,"local_action":"log_healthy"}'
```

The GET API shows stored farm records. To enable the application-to-Lambda path locally, set `FARM_LAMBDA_API_URL` to
the `ApiUrl` output and `FARM_LAMBDA_API_KEY` to the same `ObservationApiKey`. The application sends only crop/pod IDs,
confidence scores, stage labels and action metadata. If Lambda is unreachable, it continues locally and writes events to
`farm_lambda_outbox.jsonl`; its background worker retries, and any returned cloud work order is eligible to influence that
pod's next visit. A first cloud observation also creates the pod profile.

## Public or protected?

By default the GET endpoint is **public and read-only**; the POST endpoint is disabled because `ObservationApiKey` is
empty. Once you configure `ObservationApiKey`, POST requests require `x-api-key` and the local application uses that same
secret. To require a key for GET requests too, also set `ApiKey`. Keep both secrets out of source control; for anything
beyond a demo, use proper identity-based authentication and authorization.

## Changing it, and removing it

* Edit `aws/lambda/farm_api.py`, then run `python aws/sync_inline_code.py` for the read Lambda. Edit
  `aws/lambda/observation_api.py` and copy it into the `ObservationFunction` `ZipFile` block in `aws/farm_api.yaml`; both
  inline handlers must stay below CloudFormation's 4096-character limit. Validate the template before deploying.
* Remove the API: `aws cloudformation delete-stack --stack-name farm-agent-api`. The table and its data are a separate stack
  and are untouched.

## Honest status

`python dev/check_lambda_observations.py` tests the POST handler's treatment/harvest decisions, metadata-only validation,
API-key check, persistence calls, online response handling and offline outbox using local fakes (no AWS calls). The existing
`python dev/check_cloud_api.py` covers the GET API and requires the development dependencies in `dev/requirements-dev.txt`.
The template and both inline Lambda code sizes were checked locally; **a real deployment has not been tested**. The first
real checks, after the account allows the required services, are `curl "$API/health"` and a POST to `/observations`.

| Message | Meaning / fix |
|---|---|
| `{"error": "internal error"}` (500) | look in CloudWatch Logs `/aws/lambda/<function>`: usually the table name or region in `TableName` is wrong, or the table is in another region |
| `{"error": "missing or wrong x-api-key header"}` (401) | the stack was deployed with an `ApiKey`; send the header |
| `{"error": "no record for R1-L-F1"}` (404) | that pod has no record yet: run the application against this table |
| `Message: Forbidden` | the path is not one of the three routes above |
