"""
SupportIQ - Dataset Generation Module
Generates a realistic, diverse, multi-class customer support message dataset
with balanced categories, priorities, and sentiments.

Categories: billing, technical, account, shipping
Priorities: low, medium, high
Sentiments: positive, neutral, negative

Author: SupportIQ ML Engineering Team
"""

import json
import os
import random
import re
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split


# Set deterministic random seed for strict reproducibility
SEED = 42
random.seed(SEED)

CATEGORIES = ["billing", "technical", "account", "shipping"]
PRIORITIES = ["low", "medium", "high"]
SENTIMENTS = ["positive", "neutral", "negative"]

# Realistic scenario generators per category
SCENARIOS = {
    "billing": {
        "high": {
            "negative": [
                "I was charged ${amount} three separate times for order #{num}! My bank account is now in overdraft. Fix this immediately and refund the unauthorized deductions!",
                "You just debited my card without my consent for an annual renewal that I explicitly cancelled last week. This is fraudulent. Reverse the charge today!",
                "URGENT: Billed ${amount} on an enterprise account that is supposed to be on a quarterly net-30 invoice. Our accounting team halted all operations until this is resolved.",
                "Why the hell did my subscription jump from $29 to ${amount} without any warning? Refund the difference right now or I will initiate a chargeback with Amex.",
                "Emergency billing issue: your automated billing script charged my corporate credit card {num} times today. Total unauthorized charge is over ${amount}. Call me immediately!"
            ],
            "neutral": [
                "Discrepancy detected in invoice #{num}. Billed amount is ${amount} instead of agreed contract rate. Please issue an updated credit memo before payment cutoff at 5 PM.",
                "Payment gateway failed during transaction for invoice #{num}. Please hold the account suspension while we wire the payment directly.",
                "We need an immediate tax exemption certificate applied to order #{num} before the fiscal quarter closes tomorrow.",
                "Critical billing update: our primary payment method expired and automatic payroll service billing is failing today. Need phone support to process manual wire.",
                "Invoice #{num} has been sent to our legal department due to incorrect billing terms of ${amount}. Please rectify this prior to COB."
            ],
            "positive": [
                "I know your billing team is swamped, but you guys have always been fair. I was accidentally double billed ${amount} for invoice #{num}, could you please help me reverse it?",
                "Great platform as always! We are expanding our enterprise license today, but our corporate card was flagged for ${amount}. Can you help us process the invoice immediately?",
                "Super happy with the customer service so far! Just noticed a billing glitch where we were charged ${amount} twice. Hoping you can quickly process a refund.",
                "We love the service and want to upgrade immediately to the annual plan before our cycle ends tonight. Please send the link to finalize payment!",
                "Thanks for the outstanding support this quarter. We noticed an urgent duplicate charge of ${amount} on invoice #{num} and would appreciate your expedited assistance."
            ]
        },
        "medium": {
            "negative": [
                "I cancelled my subscription 3 days ago but I still received an invoice for ${amount}. Please cancel this charge and confirm.",
                "Why was I charged a late fee of ${amount} when my payment went through on the due date? Please credit this back to my account.",
                "Your checkout page applied the wrong exchange rate for my purchase of ${amount}. That cost me extra fees.",
                "I asked for an itemized receipt two days ago and still haven't received it. My company expense report is due.",
                "The discount code SAVE20 was not applied to my cart even though it said valid. I was charged the full ${amount}."
            ],
            "neutral": [
                "Can you please provide the VAT invoice for my recent payment of ${amount} on invoice #{num}?",
                "I need to change the billing currency from USD to EUR for our company profile.",
                "Please update the billing address on file for invoice #{num} to 452 Innovation Blvd.",
                "When will the refund of ${amount} processed on Monday reflect in my bank account?",
                "Where can I download all monthly statements for the year 2025 in PDF format?"
            ],
            "positive": [
                "Thank you for the prompt service. Could you please send over the official receipt for invoice #{num} when convenient?",
                "Really appreciate your clear pricing structure. Just inquiring if there is any non-profit discount available for our upcoming renewal.",
                "The transition to the team plan was super smooth. Could you assist with updating the billing email to accounting@example.com?",
                "Thanks team! We just settled invoice #{num} and wanted to confirm everything is in order on your end.",
                "Enjoying the product very much. Wondering if we can switch from monthly to annual billing to take advantage of the discount."
            ]
        },
        "low": {
            "negative": [
                "Minor issue, but the receipt layout is hard to read and doesn't clearly display the subtotal.",
                "Your pricing page is slightly confusing regarding whether sales tax is included in the monthly price.",
                "Noticed a small discrepancy of a few cents on the monthly conversion rate. Not a big deal but wanted to inform you.",
                "I wish you supported PayPal alongside credit cards; it would make checkout less tedious.",
                "The invoice email went to my spam folder because of the sender domain settings."
            ],
            "neutral": [
                "Could you clarify the billing date for next month? Is it the 1st or the 15th?",
                "Do you provide a formal receipt for transactions under $10?",
                "Just checking what payment methods you currently accept for personal plans.",
                "Is it possible to add a purchase order number to future automated invoices?",
                "What is the policy regarding prorated refunds if a user downgrades mid-cycle?"
            ],
            "positive": [
                "Love the clear billing breakdown on your new invoices! Keep up the good work.",
                "Just wanted to leave a quick note saying your refund process was the smoothest I have experienced.",
                "Kudos to the billing team for keeping everything straightforward and transparent.",
                "Great service! Just wanted to confirm that our automatic renewal discount was successfully registered.",
                "Everything looks wonderful on our latest statement. Thanks for the seamless billing experience!"
            ]
        }
    },
    "technical": {
        "high": {
            "negative": [
                "PRODUCTION CRASH: The core service is throwing 500 Internal Server Error across all endpoints since 10 minutes ago! Our entire user base is blocked!",
                "Critical data loss bug: after the latest mobile app update v4.2, all local client drafts were wiped out! Restore our database backups immediately!",
                "Our production API keys are failing authentication continuously. All background webhooks have stopped. This is causing significant revenue loss every minute!",
                "The mobile app crashes immediately on launch on all iOS 18 devices. Customers cannot access their accounts. This needs a hotfix right now!",
                "System memory leak in v2.4.1 is causing our worker containers to OOM kill every 20 minutes. We need emergency senior engineering support."
            ],
            "neutral": [
                "Critical incident: database replication lag exceeded 15 minutes on cluster-east-01. Read queries are returning stale transactional data.",
                "SSO SAML handshake is failing with error code CERT_EXPIRED on your identity provider endpoint. All corporate users are unable to authenticate.",
                "Webhook delivery endpoint is receiving HTTP 504 gateway timeouts from your server cluster. Urgent diagnostics needed.",
                "The primary WebSocket connection drops every 45 seconds under load. Trace logs attached for your infrastructure team.",
                "SSL certificate on api.example.com appears invalid or expired as of 06:00 UTC. System connections are failing handshake."
            ],
            "positive": [
                "We know how hard high-severity outages are on the on-call team. Our production API is down with code 503, cheering you on to get it sorted quickly!",
                "You guys usually have 99.99% uptime, which is why we trust you. Right now our main pipeline is blocked by an unhandled exception in the parser. Appreciate your prompt investigation!",
                "Huge fans of your platform! Unfortunately our staging environment hit a critical deadlock after today's release. Excited to see this resolved.",
                "Your incident response has always been stellar. Alerting you that our webhook listener has dropped all connections. Thanks for your quick help!",
                "Always love the rapid response from your tech support. We have a severe server connection failure on production node #{num}. Thanks in advance for the fix!"
            ]
        },
        "medium": {
            "negative": [
                "The export to CSV button is broken on the analytics dashboard. Clicking it does nothing and the browser console shows an uncaught TypeError.",
                "Search filter by date range is not returning results for records created after October 1st. Please investigate.",
                "The desktop app consumes 100% CPU whenever a report is generating in the background. My computer overheats.",
                "Push notifications have stopped arriving on Android devices since the patch yesterday.",
                "Dark mode toggle resets to light mode every time the page refreshes. It's frustrating to re-enable it constantly."
            ],
            "neutral": [
                "What is the recommended timeout setting for the bulk upload REST API to prevent connection termination?",
                "We are observing intermittent 429 Too Many Requests errors despite staying under our 100 req/min rate limit.",
                "Can you check if there is an issue with the EU-West datacenter? Latency has increased from 40ms to 280ms.",
                "The syntax highlighter does not format YAML snippets properly in the code block preview.",
                "Need technical documentation on how to configure mTLS for the webhook callback URLs."
            ],
            "positive": [
                "The new API response times are fantastic! We just ran into a small query parameter issue when paginating results beyond page 50.",
                "Really digging the new UI update! Just noticed a minor rendering bug where the navigation bar overlaps modal popups on Safari.",
                "Great work on the latest SDK release. Found a small typing bug in TypeScript definitions for the user payload.",
                "Appreciate the continuous improvements. The sync is working much faster now, just checking why one specific file format (.tsv) throws a parse warning.",
                "Thanks for building such an intuitive developer portal. Found a minor typo in the Python code example on page 4."
            ]
        },
        "low": {
            "negative": [
                "Minor annoyance: the tooltips on the settings page flicker when hovering over them quickly.",
                "The favicon appears blurry on high-DPI retina displays.",
                "Font size in the code viewer is slightly smaller than the rest of the application interface.",
                "Scroll bar on the sidebar disappears unexpectedly when scrolling through long lists.",
                "The keyboard shortcut Ctrl+K opens the search bar, but Esc doesn't always close it on Firefox."
            ],
            "neutral": [
                "Is there an open roadmap for supporting ARM64 Linux binaries for your CLI tool?",
                "Which versions of Node.js are officially supported by the current client library?",
                "Just curious if you plan to add syntax support for Rust in the markdown preview.",
                "Could you specify the maximum image upload dimensions allowed in the profile editor?",
                "Does the current desktop build support Apple Silicon natively or via Rosetta 2?"
            ],
            "positive": [
                "The app performance is silky smooth! Just wanted to share a tiny suggestion to add a dark theme option to the code viewer.",
                "Your API documentation is by far the cleanest I have worked with this year. Keep up the brilliant engineering!",
                "Loving the keyboard shortcuts feature! Makes navigating the dashboard such a breeze.",
                "Huge shoutout to the dev team for fixing the memory footprint on the latest version. It runs like a dream now.",
                "Just wanted to say thanks for maintaining such clean SDK releases. Made our team's integration super painless!"
            ]
        }
    },
    "account": {
        "high": {
            "negative": [
                "ACCOUNT COMPROMISED: I received an email saying my login password and recovery email were changed from an unrecognized IP in another country! Lock my account immediately!",
                "I am completely locked out of my master administrative account and the 2FA authenticator app is not accepting my security codes. We cannot manage our team!",
                "Our company founder was locked out right before an investor pitch due to 'Suspicious Activity Detected'. We need immediate manual override and access restoration!",
                "An unauthorized employee was able to elevate their privileges to Owner and revoke admin rights. This is a severe security violation. Intervene immediately!",
                "My account has been suspended without any explanation, and our entire enterprise workspace is inaccessible. We need this unlocked right now!"
            ],
            "neutral": [
                "Admin access token revoked unexpectedly. Please re-verify domain ownership for organization org_{num} to restore team credentials.",
                "The primary 2FA phone number on file is deactivated. Need to initiate identity verification protocol to update the multi-factor device.",
                "Urgent account audit request: please provide the complete access log and IP history for user ID #{num} for the last 48 hours.",
                "Enterprise SAML SSO configuration is preventing all users from signing in. Need administrator reset to username/password fallback.",
                "Organization owner has left the company. We require legal transfer of the primary tenant account to the new chief executive."
            ],
            "positive": [
                "I know security protocols are strict for good reason, but I'm locked out of my corporate account and need quick assistance to recover it today!",
                "Your security team did a great job flagging suspicious activity! My account was temporarily frozen for safety; could you help me verify my identity to restore access?",
                "Really appreciate your fast account security alerts! I received an alert and want to quickly reset my master 2FA keys with your help.",
                "Thank you for keeping our data safe. We are urgently onboarding 50 new staff members today and need our account seat quota refreshed.",
                "Always had great experiences with your support. Currently locked out of my account due to a lost phone; looking forward to getting verified quickly."
            ]
        },
        "medium": {
            "negative": [
                "I've requested a password reset email four times over the past hour and haven't received a single message. Checked spam already.",
                "Trying to change my registered email address, but the confirmation link keeps giving an 'Invalid or Expired Token' error.",
                "I removed a former team member from our workspace, but they still appear as active in the user list and seat count.",
                "Why can't I delete my account from the user settings? The delete button is greyed out without any explanation.",
                "The profile picture upload keeps failing with 'Upload Error' even though the JPG image is only 500 KB."
            ],
            "neutral": [
                "How do I merge two individual accounts under a single corporate workspace without losing project history?",
                "Please update the organization name on our account profile from TechCorp LLC to TechCorp Global Inc.",
                "Can you confirm whether deleting a user profile permanently removes their activity history from workspace logs?",
                "What is the procedure for transferring ownership of a project workspace to an external collaborator?",
                "I need to export our team's complete personal profile data in accordance with GDPR compliance."
            ],
            "positive": [
                "Thanks for the great customer service. Could you guide me on how to invite three new members to our team workspace?",
                "Loving the collaborative tools! Just looking for advice on how to set up role-based permissions for our contractors.",
                "The new team management portal is so much more intuitive. Just had a quick question regarding workspace branding options.",
                "Really appreciate the quick replies from your support team. Wanted to verify if our account email change went through successfully.",
                "Thank you for helping us set up the account last month. Everything is going great; just wanted to add a secondary administrator."
            ]
        },
        "low": {
            "negative": [
                "Small complaint: the username character limit is 15 characters, which is a bit restrictive.",
                "The notification settings page has too many toggles and feels somewhat cluttered.",
                "I keep receiving marketing emails even though I unchecked the promotional newsletter box in preferences.",
                "It took me a while to find where the logout button was hidden inside the submenu.",
                "Wish the avatar crop tool allowed custom aspect ratios instead of just a circle."
            ],
            "neutral": [
                "Can a single user belong to more than five distinct workspaces simultaneously?",
                "Is it possible to customize the notification digest from weekly to monthly?",
                "How do I change my display handle from lowercase to uppercase?",
                "Does changing my profile timezone automatically alter timestamp displays across reports?",
                "Where can I review the active browser sessions currently logged into my profile?"
            ],
            "positive": [
                "Love how easy it is to manage profile settings in the new dashboard!",
                "Just wanted to compliment your team on the clean and user-friendly onboarding flow.",
                "The account security features you offer give our team great peace of mind. Thanks!",
                "Super seamless setup process. Took me less than two minutes to get my profile configured.",
                "Really pleased with how straightforward account management is on your platform!"
            ]
        }
    },
    "shipping": {
        "high": {
            "negative": [
                "URGENT: Package #{num} was marked as 'Delivered on Porch' an hour ago, but nothing is here! Security cameras show no delivery driver arrived! This is a $1,200 laptop!",
                "My emergency medical equipment order #{num} was supposed to arrive overnight via Express, but the tracker shows it was sent to the wrong distribution hub across the country!",
                "Critical shipment #{num} containing event materials for our conference tomorrow morning has not arrived. Tracking hasn't updated in 4 days. Intervene immediately!",
                "The courier delivered a crushed, soaking wet box for order #{num}. The fragile glass contents inside are completely shattered! I need a replacement dispatched immediately!",
                "Order #{num} has been stuck in customs for two weeks because your shipping department failed to attach the commercial invoice! Fix this clearance issue today!"
            ],
            "neutral": [
                "Customs border agency has placed an indefinite hold on consignment #{num} pending export declaration documentation. Please provide the required manifest.",
                "Consignment #{num} is scheduled for urgent delivery today, but the destination address has a typographical error in the postal code. Correct this with the carrier immediately.",
                "Carrier tracking reports shipment #{num} was returned to sender due to 'Address Incomplete'. We need expedited reshipment to the corrected location.",
                "Perishable goods in transit on tracking number #{num} have exceeded the 48-hour cold chain window. Expedite priority courier inspection.",
                "Freight consignment #{num} required a liftgate at delivery but the driver arrived in a standard truck and refused delivery. Reschedule priority freight with liftgate."
            ],
            "positive": [
                "You guys are usually lightning fast with deliveries! Our urgent package #{num} seems delayed in transit. Hoping your amazing team can help speed it up for our event!",
                "Always love ordering from you! I have a high-priority shipment #{num} that needs an urgent delivery address adjustment before it leaves the warehouse. Thanks for being so responsive!",
                "Thank you for the prompt dispatch! The package #{num} is critical for our team demo tomorrow; could you help ensure priority courier delivery?",
                "Appreciate your reliable service as always. Package #{num} had a delivery exception notice; excited for your team to help resolve it quickly!",
                "Thanks so much for your great customer care. We have an urgent order #{num} that we'd love to upgrade to priority overnight shipping if still possible."
            ]
        },
        "medium": {
            "negative": [
                "The estimated delivery date for order #{num} has slipped three times now with no explanation from the carrier.",
                "The tracking link provided in the confirmation email leads to a 404 page on the courier's website.",
                "I paid extra for 2-day express shipping, but the tracking details show standard ground transit taking 6 days.",
                "Package #{num} was left out in the rain by the courier despite explicit instructions to place it inside the parcel locker.",
                "Only one item out of three arrived in the parcel for order #{num}. The packing slip shows all three were supposed to be included."
            ],
            "neutral": [
                "Can you provide the carrier name and tracking number for order #{num} placed last Thursday?",
                "How do I redirect package #{num} to a nearby pickup depot instead of home delivery?",
                "Is signature upon delivery required for the upcoming shipment of order #{num}?",
                "What is the average transit time for standard domestic shipping to zip code 94107?",
                "Has order #{num} been handed over to the postal service or is it still being packed at the warehouse?"
            ],
            "positive": [
                "Thanks for the prompt shipment! Just wondering if you can estimate what time of day the courier typically delivers to residential areas.",
                "Really pleased with how quickly order #{num} was processed. Could you send me the tracking link when it becomes available?",
                "The packaging is always so eco-friendly and neat! Just checking on the estimated arrival date for our latest order.",
                "Thank you for the update! Glad to hear the package is on its way. Let me know if there are any tracking updates.",
                "Love your products! Just a quick inquiry to see if package #{num} can be held at the local branch for pickup."
            ]
        },
        "low": {
            "negative": [
                "The cardboard box used for shipping was slightly oversized for such a small item. Lots of unnecessary packing peanuts.",
                "The tracking email notification arrived several hours after the package was already delivered.",
                "The shipping label was slapped right over the product instructions on the box.",
                "Would be nice if the courier sent an SMS reminder in addition to the email notification.",
                "The adhesive tape on the outer shipping box was peeling at one corner, though the item was intact."
            ],
            "neutral": [
                "Do you offer free shipping on international orders over a certain threshold?",
                "Which courier services do you partner with for standard regional deliveries?",
                "Can I opt out of physical paper receipts being included inside the shipping package?",
                "Are shipping costs refundable if an item is returned within the 30-day window?",
                "Do you ship to military APO/FPO addresses?"
            ],
            "positive": [
                "My package arrived two days earlier than expected! Fantastic delivery service as always.",
                "Huge kudos on the super fast shipping! Everything was packed securely and arrived in pristine condition.",
                "Just wanted to say thank you for the wonderful shipping experience. Really impressed!",
                "Delighted with the speedy turnaround from purchase to delivery. Excellent job!",
                "The delivery driver was so polite and placed the parcel safely behind the planter. Great service!"
            ]
        }
    }
}

# Stylistic modifiers to introduce realistic variations
TYPO_MAP = {
    "received": ["recieved", "recived"],
    "subscription": ["subcription", "subscrption"],
    "cannot": ["cant", "can't", "canno"],
    "delivery": ["delivry", "delvery"],
    "account": ["acount", "accnt"],
    "password": ["pasword", "psswd"],
    "please": ["plz", "pls"],
    "immediately": ["immediatly", "imediatly"],
    "urgent": ["urgnt", "URGENT"],
    "refund": ["refnd", "refud"],
    "service": ["servce", "srvice"],
    "issue": ["isue", "issu"]
}

CASING_STYLES = ["normal", "all_lower", "informal_caps", "expressive"]

SENTENCE_STARTERS = {
    "positive": [
        "Hey there! ", "Hi team, ", "Good morning! ", "Hello, ", "Hope you are doing great! ",
        "Quick note: ", "First off, great work! ", "Hello support crew, "
    ],
    "neutral": [
        "Hello, ", "Hi, ", "Inquiry: ", "Regarding: ", "Support request: ",
        "Good day, ", "Attention Support: ", ""
    ],
    "negative": [
        "Listen, ", "Hello? ", "Unacceptable: ", "Extremely frustrated here. ",
        "Can someone explain this? ", "Attention: ", "Hey, ", ""
    ]
}

SENTENCE_ENDERS = {
    "positive": [
        " Thanks so much!", " Appreciate your help as always.", " Have a wonderful day!",
        " Best regards.", " Cheers!", " Many thanks."
    ],
    "neutral": [
        " Awaiting your response.", " Please advise.", " Thank you.",
        " Regards.", " Let me know.", " Thanks."
    ],
    "negative": [
        " Fix this asap.", " Please resolve this immediately.", " Not happy about this.",
        " Waiting for a quick reply.", " This is ridiculous.", " Resolve this now!"
    ]
}


def inject_realistic_noise(text: str, noise_prob: float = 0.25) -> str:
    """Injects realistic spelling variations, informal shorthand, and casing."""
    words = text.split()
    new_words = []
    for word in words:
        clean_word = re.sub(r'[^\w]', '', word.lower())
        if clean_word in TYPO_MAP and random.random() < noise_prob:
            typo = random.choice(TYPO_MAP[clean_word])
            # retain punctuation if any
            if word.endswith(('.', '!', '?', ',')):
                word = typo + word[-1]
            else:
                word = typo
        new_words.append(word)
    
    noisy_text = " ".join(new_words)
    
    # Random casing variations (10% all lowercase, 5% high-caps for urgent/negative)
    style = random.random()
    if style < 0.10:
        noisy_text = noisy_text.lower()
    elif style > 0.95 and "URGENT" in noisy_text.upper():
        noisy_text = noisy_text.upper()
        
    return noisy_text


def generate_synthetic_dataset(target_size: int = 1200) -> List[Dict]:
    """
    Generates a balanced dataset of at least target_size examples.
    Covers all 4 categories, 3 priorities, and 3 sentiments (36 combinations).
    """
    combinations = [
        (cat, prio, sent)
        for cat in CATEGORIES
        for prio in PRIORITIES
        for sent in SENTIMENTS
    ]
    
    samples_per_combo = target_size // len(combinations) + 5
    dataset = []
    seen_texts = set()
    sample_id = 1

    for cat, prio, sent in combinations:
        templates = SCENARIOS[cat][prio][sent]
        count = 0
        
        while count < samples_per_combo:
            base_template = random.choice(templates)
            
            # Format realistic placeholders
            amount = str(random.choice([15, 29, 49, 99, 149, 299, 499, 1200, 2450]))
            num = str(random.randint(10000, 99999))
            text = base_template.replace("${amount}", f"${amount}").replace("{num}", num)
            
            # Add prefix / suffix variety
            if random.random() < 0.6:
                prefix = random.choice(SENTENCE_STARTERS[sent])
                suffix = random.choice(SENTENCE_ENDERS[sent])
                text = f"{prefix}{text}{suffix}".strip()
                
            # Inject realistic human typing noise
            text = inject_realistic_noise(text, noise_prob=0.20)
            
            # Clean duplicate whitespaces
            text = " ".join(text.split())
            
            # Exact deduplication check
            norm_key = text.lower()
            if norm_key in seen_texts or len(text) < 15:
                continue
                
            seen_texts.add(norm_key)
            
            output_obj = {
                "category": cat,
                "priority": prio,
                "sentiment": sent
            }
            
            record = {
                "id": f"supp_{sample_id:04d}",
                "instruction": "Classify the customer support message into JSON with category, priority, and sentiment.",
                "input": text,
                "output": json.dumps(output_obj),
                "category": cat,
                "priority": prio,
                "sentiment": sent
            }
            
            dataset.append(record)
            sample_id += 1
            count += 1

    random.shuffle(dataset)
    return dataset


def split_and_save_dataset(dataset: List[Dict], data_dir: str):
    """
    Performs a 80/10/10 stratified split and saves train, validation, test jsonl files.
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # Create composite stratification label (category_priority_sentiment)
    stratify_labels = [f"{d['category']}_{d['priority']}_{d['sentiment']}" for d in dataset]
    
    # 80% train, 20% temp (for val + test)
    train_data, temp_data, train_strat, temp_strat = train_test_split(
        dataset,
        stratify_labels,
        test_size=0.20,
        random_state=SEED,
        stratify=stratify_labels
    )
    
    # Split temp into 50% val and 50% test (10% and 10% of total)
    val_data, test_data = train_test_split(
        temp_data,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_strat
    )
    
    splits = {
        "train.jsonl": train_data,
        "validation.jsonl": val_data,
        "test.jsonl": test_data
    }
    
    for filename, split_records in splits.items():
        filepath = os.path.join(data_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            for record in split_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Saved {len(split_records)} records to {filepath}")


def main():
    print("=" * 60)
    print("SupportIQ - Dataset Generator")
    print("=" * 60)
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(base_dir, "data")
    
    print(f"Generating synthetic support messages (target >= 1,200)...")
    dataset = generate_synthetic_dataset(target_size=1200)
    print(f"Total unique records generated: {len(dataset)}")
    
    split_and_save_dataset(dataset, data_dir)
    print("Dataset generation and stratified splitting complete!")


if __name__ == "__main__":
    main()

