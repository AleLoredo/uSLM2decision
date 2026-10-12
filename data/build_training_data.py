"""
OpenJEV Multi-Task Balanced Dataset Generator (System 1)
Generates high-quality balanced datasets for OpenJEV's three canonical primitives:
1. `noul` (Boolean truth/urgency evaluation: targets 0 or 1)
2. `choice` (Categorical routing over criteria map: target option ID)
3. `score` (Ordinal rubric scoring: target discrete level integer)

Stores results in:
  ./train-dataset/
  ├── train.jsonl
  ├── val.jsonl
  ├── test.jsonl
  └── dataset_card.json
"""

import os
import json
import random
from typing import Dict, List, Any

# Ensure reproducible generation
RNG = random.Random(42)

DATASET_DIR = "train-dataset"

# ---------------------------------------------------------
# 1. NOUL (Boolean / Is this true? / Urgency / Cancellation)
# ---------------------------------------------------------

NOUL_URGENT_POSITIVES = [
    "you idiots i need this fixed by today",
    "our payment gateway is returning 500 errors on every request, we cannot process customer orders",
    "emergency: production database is locked and customer checkout is completely down",
    "I am losing sales right now, please respond immediately ASAP",
    "my account was compromised and unauthorized charges are occurring right now",
    "critical bug in production API affecting all enterprise customers",
    "we are in the middle of our product launch and the integration is failing, urgent help needed",
    "system outage: all webhooks stopped firing 15 minutes ago, critical priority",
    "help! our entire team is locked out of the dashboard during customer hours",
    "billing double charged our credit card 10 times in 5 minutes, stop this right now",
    # Spanish colloquial & high urgency
    "Son unos soretes. Dene de baja de su servicio de mierda ya",
    "Es una emergencia crítica, la base de datos se cayó y estamos perdiendo ventas ahora mismo",
    "Nos están cobrando de más y nadie responde, atiendan ya urgente",
    "Se cayó todo el sistema en producción, necesitamos ayuda inmediata",
    "Me hackearon la cuenta y están haciendo cargos no autorizados ahora mismo",
    "Urgente por favor, no podemos procesar pagos de clientes hace 2 horas",
    "Arreglen esto hoy mismo, es completamente inaceptable y urgente"
]

NOUL_URGENT_NEGATIVES = [
    "hi, just wondering if you support dark mode on mobile?",
    "no rush at all, just wanted to ask when the next feature release is planned",
    "great product! one small typo in the documentation on page 3",
    "when you have time next week, could you clarify your pricing tiers?",
    "inquiring about whether you have an affiliate partnership program",
    "just sharing some feedback, everything is working smoothly",
    "curious if you will support Python 3.14 in the upcoming quarter",
    "hello team, hope you are having a nice weekend, quick question about receipt formatting",
    "would it be possible to eventually reduce the cost of my active subscription to your service?",
    "minor suggestion: it would be nice to have a search shortcut",
    "everything is working fine, just wanted to confirm my renewal date next month",
    # Spanish low urgency / polite
    "¿Sería posible consultar por una reducción en el costo de mi suscripción activa?",
    "Hola, una consulta cuando tengan tiempo sobre los planes disponibles",
    "Buenas tardes, quería consultar si tienen soporte para modo oscuro",
    "Todo funcionando excelente por aquí, muchas gracias por el servicio",
    "Una sugerencia menor: estaría bueno agregar un atajo de teclado",
    "Sin apuro, queríamos saber la fecha de renovación del próximo mes",
    "Hola equipo, buen fin de semana, una duda rápida sobre el formato de facturas"
]

NOUL_CANCEL_POSITIVES = [
    "please cancel my subscription immediately, I do not want to be charged again",
    "I want to terminate my account and delete all associated data",
    "cancel my plan right now and issue a refund for this month",
    "close my account, I am switching to another provider",
    "stop auto-renewal and cancel my membership",
    # Spanish cancellation
    "Son unos soretes. Dene de baja de su servicio de mierda ya",
    "Quiero dar de baja mi suscripción inmediatamente, no me cobren más",
    "Por favor cancelen mi cuenta y eliminen todos mis datos personales",
    "Solicito la baja del servicio y el reembolso total de este mes",
    "Denme de baja ya mismo, me paso a otra empresa",
    "Quiero rescindir el contrato y cerrar la cuenta definitivamente",
    "Detengan la renovación automática y cancelen mi membresía ahora"
]

NOUL_CANCEL_NEGATIVES = [
    "how do I upgrade my account to the enterprise plan?",
    "we would like to add 15 more user seats to our current subscription",
    "can we renew our annual contract for another two years?",
    "where can I download the invoice for our renewed subscription?",
    "we love the tool and want to expand it to our entire engineering team",
    # Spanish renewal / upgrade
    "¿Cómo hago para renovar mi suscripción por un año más?",
    "Queremos agregar 15 licencias más a nuestro plan actual",
    "Excelente servicio, queremos ampliar el contrato corporativo por dos años",
    "¿Dónde descargo la factura de la renovación anual de nuestra cuenta?",
    "Nos encanta la herramienta y queremos extenderla a todo el equipo"
]

# ---------------------------------------------------------
# 2. CHOICE (Which of these options?)
# ---------------------------------------------------------

CHOICE_CRITERIA_DEPARTMENTS = {
    "billing": "Invoices, payments, refunds, subscription cancellations",
    "technical": "Bugs, outages, login issues",
    "sales": "Pricing, new plans, enterprise inquiries"
}

CHOICE_SAMPLES_DEPARTMENT = [
    # Billing (English)
    ("there is a problem with my billling. you idiots i need this fixed by today", "billing"),
    ("My credit card was charged twice for the monthly subscription.", "billing"),
    ("Can I get an updated VAT invoice for our accounting team?", "billing"),
    ("Where can I update our company bank details and payment method?", "billing"),
    ("Why did the subscription renew at a higher rate without notice?", "billing"),
    ("We need a refund for the unused seats from last quarter.", "billing"),
    ("Payment failed with error card_declined, how do we retry?", "billing"),
    ("Our account shows past due even though we wired payment yesterday.", "billing"),
    ("Please cancel my subscription immediately and stop charging me.", "billing"),
    
    # Billing (Spanish)
    ("Son unos soretes. Dene de baja de su servicio de mierda ya", "billing"),
    ("Quiero dar de baja mi servicio y que me devuelvan el dinero cobrado.", "billing"),
    ("Me cobraron dos veces la suscripción este mes en la tarjeta de crédito.", "billing"),
    ("Necesito la factura con el CUIT y detalle de IVA de este período.", "billing"),
    ("¿Cómo cambio el método de pago registrado para la facturación mensual?", "billing"),
    ("El pago fue rechazado con error tarjeta vencida, ¿cómo reintento?", "billing"),
    ("Quiero cancelar mi suscripción inmediatamente, no me renueven más.", "billing"),
    ("Solicito el reintegro por los días que el servicio no estuvo disponible.", "billing"),
    ("Por favor cancelen mi cuenta y detengan cualquier cargo futuro.", "billing"),
    ("Nuestra cuenta figura como vencida pero transferimos el pago ayer.", "billing"),
    
    # Technical (English)
    ("Our API integration started returning 500 errors on every request 20 minutes ago.", "technical"),
    ("The webhook endpoint is throwing 502 Bad Gateway intermittently.", "technical"),
    ("I am getting a TypeError: NoneType object is not callable in the Python SDK.", "technical"),
    ("Docker container exits immediately with exit code 137 OOM killed.", "technical"),
    ("How do I configure mutual TLS for the websocket connection?", "technical"),
    ("The database migration script failed on alter table command.", "technical"),
    ("Requests are timing out after 30 seconds when querying the search API.", "technical"),
    ("Our SSL certificate seems to have expired on the API gateway.", "technical"),

    # Technical (Spanish)
    ("La API está respondiendo con código 500 en todas las peticiones desde hace 20 minutos.", "technical"),
    ("Tenemos un error de timeout al consultar el endpoint de decisiones.", "technical"),
    ("El webhook no se está ejecutando cuando recibimos eventos.", "technical"),
    ("Error de conexión SSL en el gateway de producción.", "technical"),
    ("El contenedor de Docker se cierra con error 137 de falta de memoria OOM.", "technical"),
    ("La migración de la base de datos falló con un error en alter table.", "technical"),
    ("Obtengo TypeError en el SDK de Python al inicializar el cliente.", "technical"),

    # Sales (English)
    ("What are your enterprise pricing tiers for 250 active agents?", "sales"),
    ("We would like to request a formal RFP quote for our annual procurement.", "sales"),
    ("Do you offer volume discounts for non-profit organizations or universities?", "sales"),
    ("We want to schedule a product demo with our VP of Engineering.", "sales"),
    ("Is there an annual discount if we pay upfront instead of monthly?", "sales"),
    ("Can we get custom terms and an SLA guarantee for enterprise tier?", "sales"),
    ("How does your pricing compare with competitor X for 5 million tokens?", "sales"),
    ("would it be possible to eventually reduce the cost of my active subscription to your service?", "sales"),
    ("We want to discuss custom contract terms before signing the order form.", "sales"),

    # Sales (Spanish)
    ("¿Cuáles son los precios y planes corporativos para 250 usuarios activos?", "sales"),
    ("Quisiéramos coordinar una demo del producto con el vicepresidente de tecnología.", "sales"),
    ("¿Tienen descuento si pagamos la suscripción anual por adelantado?", "sales"),
    ("Necesitamos un presupuesto formal para compras corporativas.", "sales"),
    ("¿Tienen planes especiales para startups o instituciones educativas?", "sales"),
    ("Queremos consultar las tarifas y límites de tokens para contrato enterprise.", "sales")
]

CHOICE_CRITERIA_PRIORITY = {
    "p0_critical": "System completely down, severe revenue or data loss",
    "p1_high": "Major functionality broken with no workaround",
    "p2_medium": "Minor feature impaired, workaround available",
    "p3_low": "Cosmetic issue, feature request, or general question"
}

CHOICE_SAMPLES_PRIORITY = [
    ("All customer payments are failing in production right now!", "p0_critical"),
    ("Core service is crashing on every request across all regions.", "p0_critical"),
    ("Se cayó la base de datos central en producción, todo bloqueado.", "p0_critical"),
    ("Export to CSV is failing, but users can still view data on screen.", "p1_high"),
    ("La exportación falla pero la visualización sigue funcionando.", "p1_high"),
    ("Single user cannot reset password, but admin can reset it manually.", "p2_medium"),
    ("Un usuario puntual no puede restablecer contraseña.", "p2_medium"),
    ("Button padding looks slightly misaligned on Safari browser.", "p3_low"),
    ("Would love to see a dark mode option added in the future.", "p3_low"),
    ("Estaría bueno que agreguen un botón de modo oscuro en el futuro.", "p3_low")
]

# ---------------------------------------------------------
# 3. SCORE (Which level on rubric?)
# ---------------------------------------------------------

SCORE_RUBRIC_FRUSTRATION = [
    "Calm, just stating facts",
    "Frustrated but civil",
    "Very angry, strong language"
]

SCORE_SAMPLES_FRUSTRATION = [
    # Level 0 (Calm, just stating facts - English)
    ("would it be possible to eventually reduce the cost of my active subscription to your service?", 0),
    ("could we explore options to reduce the monthly cost of our subscription?", 0),
    ("is there any way to reduce the cost of my active subscription to your service?", 0),
    ("hello, I am writing to politely ask if we can reduce the cost of our current subscription.", 0),
    ("we would like to keep using your service, but need to reduce the subscription cost.", 0),
    ("could you review our active subscription to your service and suggest lower cost tiers?", 0),
    ("is it possible to get a discount on my active subscription to your service?", 0),
    ("I would like to ask a general question about my active subscription to your service.", 0),
    ("How do I manage my active subscription to your service?", 0),
    ("Thank you for your service, I have a quick question about my active subscription.", 0),
    ("Hi, just inquiring if there is any flexibility on the pricing for our active subscription.", 0),
    ("Would it be feasible to discuss a cost reduction for our recurring subscription plan?", 0),
    ("Hello, could you please clarify how the API rate limit is counted?", 0),
    ("Hi, our team is testing the endpoint and we have a quick question.", 0),
    ("Good morning, here is the log snippet for your review when convenient.", 0),
    ("I noticed an inconsistency in the documentation for the decide endpoint.", 0),
    ("Thank you for the quick reply, could you also confirm the port number?", 0),
    ("Hello, could you please tell me if there are discounts available on our active plan?", 0),
    ("Hi, just inquiring if we can adjust our monthly subscription tier.", 0),
    ("Could you please send me our invoice for last month?", 0),
    ("Where can I find the billing breakdown for extra user seats?", 0),
    ("Is it possible to downgrade our plan to the starter tier?", 0),
    ("How do I update our credit card details for future renewal?", 0),
    ("Can you please clarify the payment terms for the enterprise contract?", 0),
    ("Could you provide more documentation on configuring SAML SSO?", 0),
    ("We are planning our migration and have a question regarding data export.", 0),
    ("I would like to purchase the HBO pack for my account.", 0),
    ("I want to subscribe to the HBO pack and sports bundle.", 0),
    ("How do I contract the premium streaming package?", 0),
    ("We would like to buy the additional enterprise channels pack.", 0),
    ("Can I add the HBO pack to our monthly subscription?", 0),
    ("Interested in contracting the fiber optic and TV package.", 0),
    ("I would like to sign up for the sports and movie pack.", 0),
    
    # Level 0 (Calm, just stating facts - Spanish)
    ("quisiera contratar el pack HBO", 0),
    ("quisiera contratar el pack HBO y fútbol para este mes", 0),
    ("buenas tardes, quisiera contratar el pack premium de canales", 0),
    ("hola, me gustaría contratar el servicio de streaming y HBO", 0),
    ("deseo contratar el paquete de películas y series", 0),
    ("quisiera contratar una línea adicional y el pack de datos", 0),
    ("quiero contratar el servicio de internet y televisión", 0),
    ("hola, quisiera saber cómo contratar el pack HBO Max", 0),
    ("solicito contratar el pack de canales internacionales", 0),
    ("quisiera activar el pack premium en mi abono mensual", 0),
    ("me interesa contratar el plan pro y el pack adicional", 0),
    ("quisiera contratar el pack de deportes", 0),
    ("quiero contratar internet y el pack fútbol para mi domicilio", 0),
    ("quisiera contratar el servicio de telefonía y cable", 0),
    ("cómo hago para contratar el pack fútbol y HBO", 0),
    ("quisiera comprar el paquete de datos extra para mi línea", 0),
    ("deseo contratar el servicio de televisión digital con HBO", 0),
    ("quisiera solicitar el alta del servicio y el pack de canales", 0),
    ("quiero agregar el pack HBO a mi suscripción mensual", 0),
    ("hola, quisiera contratar el servicio para mi empresa", 0),
    ("quisiera contratar", 0),
    ("contratar el pack HBO", 0),
    ("quisiera contratar el paquete básico", 0),
    ("solicito información para contratar el pack HBO", 0),
    ("¿Sería posible consultar por una reducción en el costo de mi suscripción activa?", 0),
    ("Buenas tardes, quería consultar alternativas de planes para optimizar costos.", 0),
    ("Hola, una consulta sobre si es posible reducir el costo de mi suscripción activa al servicio.", 0),
    ("Quisiéramos saber si hay opciones para ajustar el valor del abono mensual sin perder funciones.", 0),
    ("Hola equipo, muchas gracias por su servicio, quería consultar por promociones vigentes en mi plan.", 0),
    ("Hola, buenos días, ¿podrían confirmarme cuál es el puerto por defecto del servicio?", 0),
    ("Hola, una consulta tranquila sobre cómo se calcula el límite de peticiones.", 0),
    ("Aquí les adjunto el log para que lo revisen cuando tengan disponibilidad, gracias.", 0),
    ("Encontré un pequeño detalle en la documentación técnica para consultar.", 0),
    ("Muchas gracias por la respuesta anterior, todo quedó muy claro y funcionando.", 0),
    ("Hola, quisiera saber si tienen descuentos para organizaciones sin fines de lucro.", 0),
    ("Quisiera consultar si podemos cambiar la fecha de cobro de la factura mensual.", 0),
    ("¿Cómo puedo descargar las facturas anteriores de mi cuenta?", 0),
    ("Hola, una consulta para cambiar de plan anual a mensual.", 0),
    ("¿Podrían enviarme el comprobante de pago de la última renovación?", 0),
    ("Hola, estamos configurando la integración y quisiéramos consultar una duda técnica.", 0),

    # Level 1 (Frustrated but civil - English)
    ("I have been trying to connect my Stripe account for 3 days and the integration keeps failing.", 1),
    ("This is the second time this week our sync has broken, please look into this.", 1),
    ("I submitted a ticket 48 hours ago and have not received any response yet.", 1),
    ("The error messages are completely unhelpful and we are losing time.", 1),
    ("We followed the docs step by step and it still does not work as described.", 1),
    ("I was promised a discount code last week but it was not applied to our invoice.", 1),
    ("Our card was charged yesterday despite our downgrade request 5 days ago, please check.", 1),
    ("I have been waiting for the billing department to get back to me regarding the overcharge.", 1),
    ("This pricing change came as an unpleasant surprise and our budget is strained.", 1),
    ("We were billed twice this month and have not received an explanation yet.", 1),
    ("I ordered the HBO pack three days ago and the channels are still not active.", 1),
    ("We requested the sports bundle upgrade on Monday but our account was not updated.", 1),
    
    # Level 1 (Frustrated but civil - Spanish)
    ("Hace 3 días que estoy intentando configurar la cuenta y sigue dando error.", 1),
    ("Es la segunda vez en la semana que se corta la sincronización, por favor revisen.", 1),
    ("Mandé un ticket hace 48 horas y todavía nadie me contestó.", 1),
    ("Seguimos todos los pasos de la guía y no funciona como dice la documentación.", 1),
    ("Los mensajes de error no ayudan en nada y estamos perdiendo bastante tiempo.", 1),
    ("Hace 3 días que estoy intentando comunicarme con facturación y no responden.", 1),
    ("Me cobraron la tarifa completa a pesar de que pedí el cambio de plan la semana pasada.", 1),
    ("El sistema sigue dando error al intentar actualizar la tarjeta, por favor solucionen esto.", 1),
    ("Esperaba una respuesta antes respecto a la devolución del dinero cobrado de más.", 1),
    ("Nos cobraron dos veces la suscripción este mes y todavía no tenemos reintegro.", 1),
    ("Pedí la contratación del pack HBO hace tres días y todavía no tengo los canales habilitados.", 1),
    ("Intenté contratar el pack fútbol dos veces pero sigue dando error en la plataforma.", 1),
    ("Solicité el alta del pack premium la semana pasada y todavía no se activó.", 1),

    # Level 2 (Very angry, strong language - English)
    ("there is a problem with my billling. you idiots i need this fixed by today", 2),
    ("this is completely unacceptable and your software is garbage!", 2),
    ("you charged us 10 times and your support is utterly useless, fix this immediately!", 2),
    ("what a joke of a company, your incompetence is costing us thousands of dollars!", 2),
    ("I want to speak with your manager immediately, this is atrocious service!", 2),
    ("You screwed up our entire billing and now our account is suspended, fix this right now!", 2),
    ("Your customer support is a disgrace, you stole our money and ghosted us!", 2),
    ("You charged me for the HBO pack and none of the channels work, you thieves, refund me now!", 2),
    
    # Level 2 (Very angry, strong language - Spanish)
    ("Son unos soretes. Dene de baja de su servicio de mierda ya", 2),
    ("Son unos ladrones y unos estafadores, me robaron la plata en la cara!", 2),
    ("Es una porquería de servicio, son unos inútiles de mierda!", 2),
    ("Esto es una vergüenza total, váyanse al carajo, exijo mi plata ya!", 2),
    ("Me cobraron cinco veces y el soporte es una basura que no sirve para nada!", 2),
    ("Manga de chantas, son un desastre, cancelen todo inmediatamente ya!", 2),
    ("Son unos incompetentes totales, me cansé de este servicio nefasto!", 2),
    ("Me cobran el pack HBO todos los meses y no anda nunca, son unos estafadores de mierda!", 2),
    ("Hace una semana que pedí el pack de canales y son un desastre, váyanse al carajo!", 2)
]

SCORE_RUBRIC_SEVERITY = [
    "Informational or trivial",
    "Minor impairment",
    "Severe disruption"
]

SCORE_SAMPLES_SEVERITY = [
    # Level 0
    ("Typo on the landing page header text.", 0),
    ("Icon color is slightly off in dark mode.", 0),
    ("Error tipográfico menor en el encabezado de la página.", 0),
    ("Documentation example has a minor broken hyperlink.", 0),
    ("El logo se ve ligeramente descentrado en pantallas pequeñas.", 0),
    
    # Level 1
    ("Notification email was delayed by 5 minutes.", 1),
    ("Search filters take 2 seconds longer than usual.", 1),
    ("El correo de notificación tardó 5 minutos en enviarse.", 1),
    ("CSV export takes several minutes to complete during peak hours.", 1),
    ("La carga del dashboard está demorando más de lo habitual.", 1),
    
    # Level 2
    ("Entire customer database is offline and requests are failing 100%.", 2),
    ("Security breach: customer tokens leaked in public logs.", 2),
    ("Toda la base de datos principal está caída y el servicio no responde.", 2),
    ("Payment processing pipeline crashed and all customer checkouts are failing.", 2),
    ("Caída total del cluster de producción, ningún endpoint responde.", 2)
]

# ---------------------------------------------------------
# Generator Functions with Data Augmentation
# ---------------------------------------------------------

def generate_noul_dataset(target_count: int = 600) -> List[Dict[str, Any]]:
    samples = []
    urg_instructions = [
        "The message conveys urgency or time-sensitivity",
        "Determine if the request is urgent",
        "Is this urgent?",
        "Urgency evaluation"
    ]
    canc_instructions = [
        "The user is requesting account or subscription cancellation",
        "Determine if the user wants to cancel",
        "Is this a cancellation request?",
        "Account cancellation intent"
    ]
    
    # 1. Urgency evaluation
    for _ in range(target_count // 4):
        q_urg = {
            "type": "noul",
            "instructions": RNG.choice(urg_instructions)
        }
        p = RNG.choice(NOUL_URGENT_POSITIVES)
        samples.append({
            "primitive": "noul",
            "state": p,
            "question": q_urg,
            "target": 1
        })
        n = RNG.choice(NOUL_URGENT_NEGATIVES)
        samples.append({
            "primitive": "noul",
            "state": n,
            "question": q_urg,
            "target": 0
        })

    # 2. Cancellation evaluation
    for _ in range(target_count // 4):
        q_canc = {
            "type": "noul",
            "instructions": RNG.choice(canc_instructions)
        }
        p = RNG.choice(NOUL_CANCEL_POSITIVES)
        samples.append({
            "primitive": "noul",
            "state": p,
            "question": q_canc,
            "target": 1
        })
        n = RNG.choice(NOUL_CANCEL_NEGATIVES)
        samples.append({
            "primitive": "noul",
            "state": n,
            "question": q_canc,
            "target": 0
        })
        
    return samples

def generate_choice_dataset(target_count: int = 600) -> List[Dict[str, Any]]:
    samples = []
    dept_instructions = [
        "Route request to the correct department",
        "Which team should handle this",
        "Select the appropriate department",
        "Determine destination team"
    ]
    prio_instructions = [
        "Select the appropriate incident severity priority",
        "Determine incident priority level",
        "Choose priority classification"
    ]
    
    # 1. Department routing
    for _ in range(target_count // 2):
        q_dept = {
            "type": "choice",
            "instructions": RNG.choice(dept_instructions),
            "criteria": CHOICE_CRITERIA_DEPARTMENTS
        }
        text, target = RNG.choice(CHOICE_SAMPLES_DEPARTMENT)
        samples.append({
            "primitive": "choice",
            "state": text,
            "question": q_dept,
            "target": target
        })

    # 2. Priority routing
    for _ in range(target_count // 2):
        q_prio = {
            "type": "choice",
            "instructions": RNG.choice(prio_instructions),
            "criteria": CHOICE_CRITERIA_PRIORITY
        }
        text, target = RNG.choice(CHOICE_SAMPLES_PRIORITY)
        samples.append({
            "primitive": "choice",
            "state": text,
            "question": q_prio,
            "target": target
        })
        
    return samples

def generate_score_dataset(target_count: int = 600) -> List[Dict[str, Any]]:
    samples = []
    frust_instructions = [
        "Determine frustration level of the user",
        "How frustrated the customer appears",
        "Rate user frustration level",
        "Evaluate customer frustration"
    ]
    sev_instructions = [
        "Rate the operational impact severity level",
        "Determine severity score",
        "Evaluate incident impact"
    ]
    
    frust_by_level = {
        0: [s for s, t in SCORE_SAMPLES_FRUSTRATION if t == 0],
        1: [s for s, t in SCORE_SAMPLES_FRUSTRATION if t == 1],
        2: [s for s, t in SCORE_SAMPLES_FRUSTRATION if t == 2]
    }
    sev_by_level = {
        0: [s for s, t in SCORE_SAMPLES_SEVERITY if t == 0],
        1: [s for s, t in SCORE_SAMPLES_SEVERITY if t == 1],
        2: [s for s, t in SCORE_SAMPLES_SEVERITY if t == 2]
    }

    # 1. Frustration rubric (balanced across levels)
    per_level_frust = (target_count // 2) // 3
    for lvl in [0, 1, 2]:
        for _ in range(per_level_frust):
            q_frust = {
                "type": "score",
                "instructions": RNG.choice(frust_instructions),
                "criteria": SCORE_RUBRIC_FRUSTRATION
            }
            text = RNG.choice(frust_by_level[lvl])
            samples.append({
                "primitive": "score",
                "state": text,
                "question": q_frust,
                "target": lvl
            })

    # 2. Severity rubric (balanced across levels)
    per_level_sev = (target_count // 2) // 3
    for lvl in [0, 1, 2]:
        for _ in range(per_level_sev):
            q_sev = {
                "type": "score",
                "instructions": RNG.choice(sev_instructions),
                "criteria": SCORE_RUBRIC_SEVERITY
            }
            text = RNG.choice(sev_by_level[lvl])
            samples.append({
                "primitive": "score",
                "state": text,
                "question": q_sev,
                "target": lvl
            })
        
    return samples

def build_all_training_data(output_dir: str = DATASET_DIR, count_per_primitive: int = 600) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    
    noul_data = generate_noul_dataset(count_per_primitive)
    choice_data = generate_choice_dataset(count_per_primitive)
    score_data = generate_score_dataset(count_per_primitive)
    
    all_records = noul_data + choice_data + score_data
    RNG.shuffle(all_records)
    
    total = len(all_records)
    train_end = int(0.80 * total)
    val_end = int(0.90 * total)
    
    train_records = all_records[:train_end]
    val_records = all_records[train_end:val_end]
    test_records = all_records[val_end:]
    
    def write_jsonl(path: str, records: List[Dict[str, Any]]):
        with open(path, "w", encoding="utf-8") as f:
            for i, r in enumerate(records):
                item = {"id": f"rec_{i:05d}", **r}
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                
    train_path = os.path.join(output_dir, "train.jsonl")
    val_path = os.path.join(output_dir, "val.jsonl")
    test_path = os.path.join(output_dir, "test.jsonl")
    
    write_jsonl(train_path, train_records)
    write_jsonl(val_path, val_records)
    write_jsonl(test_path, test_records)
    
    manifest = {
        "dataset_name": "openjev-system1-multitask",
        "version": "1.0.0",
        "output_directory": output_dir,
        "total_samples": total,
        "split": {
            "train": len(train_records),
            "val": len(val_records),
            "test": len(test_records)
        },
        "primitives": {
            "noul": len(noul_data),
            "choice": len(choice_data),
            "score": len(score_data)
        },
        "license": "Apache-2.0"
    }
    
    card_path = os.path.join(output_dir, "dataset_card.json")
    with open(card_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"✅ Dataset multi-tarea generado exitosamente en: {output_dir}/")
    print(f"   Total: {total} ejemplos (Train: {len(train_records)}, Val: {len(val_records)}, Test: {len(test_records)})")
    print(f"   Primitivas balanceadas: noul={len(noul_data)}, choice={len(choice_data)}, score={len(score_data)}")
    return manifest

if __name__ == "__main__":
    build_all_training_data()
