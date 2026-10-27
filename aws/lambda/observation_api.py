import hmac,json,os,re
from datetime import datetime,timezone
import boto3
T=boto3.resource('dynamodb').Table(os.environ['TABLE_NAME'])
KEY=os.environ.get('API_KEY','')
BAD={'image','images','frame','pixels','image_bytes','image_b64','base64'}
def hasimg(x): return any(k.lower() in BAD or hasimg(v) for k,v in x.items()) if isinstance(x,dict) else any(hasimg(v) for v in x) if isinstance(x,list) else False
TAG=re.compile(r'^R[1-3]-[LMR]-[FB][1-4]$')
HARVEST={'cabbage':'S9','lettuce':'stage 04- Harvest stage','mushroom':'Harvest'}
def out(code,data): return {'statusCode':code,'headers':{'content-type':'application/json','access-control-allow-origin':'*','cache-control':'no-store'},'body':json.dumps(data)}
def handler(event,context):
 try:
  headers={k.lower():v for k,v in (event.get('headers') or {}).items()}
  if not KEY: return out(503,{'error':'observation API key is not configured'})
  if not hmac.compare_digest(headers.get('x-api-key',''),KEY): return out(401,{'error':'unauthorized'})
  if event.get('routeKey')!='POST /observations': return out(404,{'error':'not found'})
  try: d=json.loads(event.get('body') or '{}')
  except ValueError: return out(400,{'error':'body must be JSON'})
  if not isinstance(d,dict) or hasimg(d): return out(400,{'error':'metadata only'})
  pid=d.get('pod_id',''); crop=d.get('crop_type',''); oid=d.get('observation_id','')
  if not TAG.fullmatch(pid) or crop not in HARVEST or not re.fullmatch(r'[A-Za-z0-9-]{8,80}',oid): return out(400,{'error':'bad pod/crop/event id'})
  try:
   gc=float(d.get('growth_confidence',0)); dc=float(d.get('disease_confidence',0)); visit=max(0,int(d.get('visit',0))); flag=bool(d.get('disease_flag'))
   if not 0<=gc<=1 or not 0<=dc<=1: raise ValueError()
  except (TypeError,ValueError): return out(400,{'error':'bad confidence/visit'})
  pk='POD#'+pid; obskey='OBSERVATION#'+oid
  stage=d.get('growth_stage') or ''; ts=str(d.get('timestamp') or datetime.now(timezone.utc).isoformat(timespec='seconds'))
  if flag and dc>=.5: action,task,reason='flag_for_treatment','TREAT','Disease threshold met.'
  elif not stage or gc<.4 or (flag and dc<.5): action,task,reason='schedule_frequent_monitoring','VERIFY','Low confidence; inspect again.'
  elif stage==HARVEST[crop]: action,task,reason='flag_for_harvest','HARVEST','Harvest stage detected.'
  else: action,task,reason='log_healthy','MONITOR','Monitor.'
  order={'order_id':'lambda-'+oid[:20],'pod_id':pid,'crop_type':crop,'task':task,'action':action,'status':'awaiting_approval','created':ts,'title':task,'reason':reason}
  common={'pk':pk,'pod_id':pid,'crop_type':crop,'ts':ts,'visit':visit,'growth_stage':stage,'growth_confidence':gc,'disease_flag':flag,'disease_name':(d.get('disease_name') or '') if flag else '','disease_confidence':dc}
  T.put_item(Item={**common,'sk':obskey,'entity':'CLOUD_OBSERVATION','action':action,'work_order':order})
  T.put_item(Item={**common,'sk':'INSPECTION#'+ts+'#'+str(visit).zfill(4),'entity':'INSPECTION','action':action,'reason':reason})
  T.put_item(Item={'pk':pk,'sk':'ORDER#'+ts+'#'+order['order_id'],'entity':'ORDER','pod_id':pid,'crop_type':crop,'order_id':order['order_id'],'task':task,'status':'awaiting_approval','created':ts,'updated':ts,'action':action})
  status='disease_flagged' if action=='flag_for_treatment' else 'harvest_ready' if action=='flag_for_harvest' else 'growing'
  T.update_item(Key={'pk':pk,'sk':'PROFILE'},UpdateExpression='SET entity=:e,pod_id=:p,crop_type=:c,growth_stage=:g,disease_flag=:df,disease_name=:dn,disease_confidence=:dc,last_inspection_at=:ts,last_action=:a,status=:s,gsi1pk=:ck,gsi1sk=:p,gsi2pk=:sk,gsi2sk=:gsk,updated_at=:ts',ExpressionAttributeValues={':e':'PROFILE',':p':pid,':c':crop,':g':stage,':df':flag,':dn':common['disease_name'],':dc':dc,':ts':ts,':a':action,':s':status,':ck':'CROP#'+crop,':sk':'STATUS#'+status,':gsk':ts+'#'+pid})
  return out(200,{'accepted':True,'stored':True,'source':'aws_lambda','work_order':order})
 except Exception as e:
  print('observation error',type(e).__name__,str(e)[:160]); return out(500,{'error':'internal error'})
