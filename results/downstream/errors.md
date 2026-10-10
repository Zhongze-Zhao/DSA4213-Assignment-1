# Part III representative errors (AG News test)


## GloVe-LSTM (mean) correct, random-LSTM (mean) wrong (1388 examples)

- [Sports] (lstm/mean=Sci/Tech, lstm_glove/mean=Sports; OOV 15%, 74 tok) Bekele, Isinbayeva top track athletes Names Ethiopian distance runner Kenenisa Bekele and Russian pole vaulter Yelena Isinbayeva were named male and female athletes of the year by the world track and field federation. Is
- [Business] (lstm/mean=Sci/Tech, lstm_glove/mean=Business; OOV 18%, 34 tok) Suit says secret accounts at DCX used for bribes The Securities and Exchange Commission is investigating allegations that German automaker DaimlerChrysler AG maintained at least 40 secret bank accounts to bribe foreign g
- [Sci/Tech] (lstm/mean=Business, lstm_glove/mean=Sci/Tech; OOV 10%, 50 tok) Dell unveils holiday lineup, including new plasma TVs OCTOBER 15, 2004 (IDG NEWS SERVICE) - Dell Inc. took the wraps off its holiday lineup on Thursday, showing new printers, plasma televisions and music players that wil
- [Sci/Tech] (lstm/mean=World, lstm_glove/mean=Sci/Tech; OOV 8%, 38 tok) Briefly: T-Mobile sells Sidekick II roundup Plus: Spyware bill moves to Senate...Supercomputer center gets new No. 2...MIT, Caltech offer low-tech voting advice.
- [Sci/Tech] (lstm/mean=Business, lstm_glove/mean=Sci/Tech; OOV 0%, 50 tok) One in four servers to run Linux by 2008 In a report, the research firm painted a bright future for the open source operating system, claiming that shipments of servers running Linux -- and revenues from those shipments 
- [World] (lstm/mean=Sci/Tech, lstm_glove/mean=World; OOV 13%, 53 tok) Russia agrees to Dalai Lama visit despite certain anger from China (AFP) AFP - Russia will allow Tibet's exiled spiritual leader, the Dalai Lama, to visit a southern Buddhist Russian region for the first time, the foreig

## Transformer (mean) correct, LSTM (mean) wrong (1006 examples)

- [Business] (lstm/mean=Sci/Tech, transformer/mean=Business; OOV 11%, 38 tok) HK banks to adopt different prime rates HSBC, Standard Chartered, Hang Seng bank announced Thursday that they will cut their lending and saving rates despite the 25 points rate hike in the United States overnight.
- [Business] (lstm/mean=World, transformer/mean=Business; OOV 8%, 40 tok) Options expensing delay not enough,say US senators A group of Republican senators vowed on Thursday to use the closing days of Congress this year to try and stop accounting rulemakers from requiring the expensing of stoc
- [Business] (lstm/mean=World, transformer/mean=Business; OOV 3%, 36 tok) Dollar Extends Recovery Against Euro, Yen The dollar rebounded for a second session on Thursday as traders and investors took profits on bets against the US currency before the year draws to a close.
- [Sci/Tech] (lstm/mean=Business, transformer/mean=Sci/Tech; OOV 10%, 59 tok) Suse warns of hole in Linux kernel OCTOBER 27, 2004 (TECHWORLD.COM) - Linux distributor Suse has warned of one of the most serious security holes to date in version 2.6 of the Linux kernel, which could allow attackers to
- [Business] (lstm/mean=Sci/Tech, transformer/mean=Business; OOV 6%, 31 tok) Delta Financing to Use Prepaid SkyMiles Shares of Delta Air Lines soared after the troubled airline said Monday it entered into a commitment letter with American Express Travel Related Services Co.
- [Sci/Tech] (lstm/mean=Business, transformer/mean=Sci/Tech; OOV 25%, 12 tok) Clarke takes charge of Blunkett's Fear Agenda &lt;strong&gt;Analysis&lt;/strong&gt; Horizontal drinkers rejoice

## All five LMs wrong with mean pooling (532 examples)

- [Sci/Tech] (lstm_glove/mean=Business, transformer/mean=Business; OOV 0%, 59 tok) Dell Exits Low-End China Consumer PC Market  HONG KONG (Reuters) - Dell Inc. &lt;DELL.O&gt;, the world's  largest PC maker, said on Monday it has left the low-end  consumer PC market in China and cut its overall growth t
- [World] (lstm_glove/mean=Business, transformer/mean=Business; OOV 5%, 59 tok) Stocks Climb on Strong Economic Data NEW YORK - Newly optimistic investors sent stocks sharply higher Friday, propelling the Dow Jones industrials up more than 100 points, after economic data showed strength in manufactu
- [Business] (lstm_glove/mean=Sci/Tech, transformer/mean=Sci/Tech; OOV 11%, 158 tok) Product Previews palmOneUpgrades Treo With Faster Chip, Better Display\With more than 600,000 units shipped, the Treo 600 is one of the big smartphone success stories. Last week, palmOne introduced the follow-on Treo 650
- [Sci/Tech] (lstm_glove/mean=Business, transformer/mean=Business; OOV 10%, 50 tok) Intuit Posts Wider Loss After Charge (Reuters) Reuters - Intuit Inc. (INTU.O), maker of\the No. 1 U.S. tax presentation software TurboTax, on Wednesday\posted a wider quarterly loss after taking a goodwill\impairment cha
- [Sci/Tech] (lstm_glove/mean=World, transformer/mean=World; OOV 3%, 36 tok) Exeter uni cans chemistry department In a move that has been dubbed as  #39;disastrous #39; by the Royal Society of Chemistry, Exeter University is to drop the teaching of chemistry as a subject.
- [Sci/Tech] (lstm_glove/mean=Business, transformer/mean=Business; OOV 4%, 52 tok) IBM posts broad Q3 revenue growth NEW YORK - IBM Corp. posted quarterly results on Monday showing 9 percent revenue growth from last year and slight earnings growth, despite a \$320 million charge it took during the quar

## Input-embedding mean correct but contextual mean wrong (random LSTM) (1028 examples)

- [Sci/Tech] (lstm/mean=Business, lstm/emb=Sci/Tech; OOV 7%, 58 tok) Hurricane center #39;s projection on Charley not far off, data show FORT LAUDERDALE, Fla. - (KRT) - Despite criticism that it should have better anticipated Hurricane Charley #39;s rapid intensification and quick turn, t
- [Business] (lstm/mean=World, lstm/emb=Business; OOV 9%, 35 tok) Legal Loophole Inflates Profits in Student Loans The White House could have closed a loophole through which student loan companies are billing the federal government nearly a billion dollars, but chose not to.
- [Business] (lstm/mean=Sci/Tech, lstm/emb=Business; OOV 11%, 46 tok) Belarus Bank Denies Money Laundering Charge A bank in Belarus has denied US charges that it laundered money for former Iraqi leader Saddam Hussein. Infobank, in a statement, said it has strictly followed international ag
- [Sci/Tech] (lstm/mean=Business, lstm/emb=Sci/Tech; OOV 11%, 66 tok) Sun acquisition to boost IT services portfolio Sun Microsystemshas agreed to purchase Ashburn, Virginia, IT services company SevenSpace, the companies announced Monday. With the purchase, Sun takes a further step away fr
- [Sci/Tech] (lstm/mean=Business, lstm/emb=Sci/Tech; OOV 2%, 50 tok) Authorities shut down UK-based news Web sites US authorities, participating in an international investigation, have shut down 20 independent news Web sites run by the Independent Media Center (Indymedia) by seizing two U
- [World] (lstm/mean=Sports, lstm/emb=World; OOV 16%, 32 tok) Karzai heading for Afghan victory Supporters of Afghan President Hamid Karzai say he is on course to win the presidential elections, with about one-quarter of the votes counted.
