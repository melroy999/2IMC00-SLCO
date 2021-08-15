/**
 */
package org.xtext.textualSlco.impl;

import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.notify.NotificationChain;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;

import org.eclipse.emf.ecore.impl.ENotificationImpl;

import org.xtext.textualSlco.PortReference;
import org.xtext.textualSlco.TextualSlcoPackage;
import org.xtext.textualSlco.UnidirectionalChannel;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Unidirectional Channel</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link org.xtext.textualSlco.impl.UnidirectionalChannelImpl#getSourceObject <em>Source Object</em>}</li>
 *   <li>{@link org.xtext.textualSlco.impl.UnidirectionalChannelImpl#getSourcePort <em>Source Port</em>}</li>
 *   <li>{@link org.xtext.textualSlco.impl.UnidirectionalChannelImpl#getTargetObject <em>Target Object</em>}</li>
 *   <li>{@link org.xtext.textualSlco.impl.UnidirectionalChannelImpl#getTargetPort <em>Target Port</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class UnidirectionalChannelImpl extends ChannelImpl implements UnidirectionalChannel
{
  /**
   * The cached value of the '{@link #getSourceObject() <em>Source Object</em>}' reference.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see #getSourceObject()
   * @generated
   * @ordered
   */
  protected org.xtext.textualSlco.Object sourceObject;

  /**
   * The cached value of the '{@link #getSourcePort() <em>Source Port</em>}' containment reference.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see #getSourcePort()
   * @generated
   * @ordered
   */
  protected PortReference sourcePort;

  /**
   * The cached value of the '{@link #getTargetObject() <em>Target Object</em>}' reference.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see #getTargetObject()
   * @generated
   * @ordered
   */
  protected org.xtext.textualSlco.Object targetObject;

  /**
   * The cached value of the '{@link #getTargetPort() <em>Target Port</em>}' containment reference.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see #getTargetPort()
   * @generated
   * @ordered
   */
  protected PortReference targetPort;

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  protected UnidirectionalChannelImpl()
  {
    super();
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  @Override
  protected EClass eStaticClass()
  {
    return TextualSlcoPackage.Literals.UNIDIRECTIONAL_CHANNEL;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public org.xtext.textualSlco.Object getSourceObject()
  {
    if (sourceObject != null && sourceObject.eIsProxy())
    {
      InternalEObject oldSourceObject = (InternalEObject)sourceObject;
      sourceObject = (org.xtext.textualSlco.Object)eResolveProxy(oldSourceObject);
      if (sourceObject != oldSourceObject)
      {
        if (eNotificationRequired())
          eNotify(new ENotificationImpl(this, Notification.RESOLVE, TextualSlcoPackage.UNIDIRECTIONAL_CHANNEL__SOURCE_OBJECT, oldSourceObject, sourceObject));
      }
    }
    return sourceObject;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public org.xtext.textualSlco.Object basicGetSourceObject()
  {
    return sourceObject;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public void setSourceObject(org.xtext.textualSlco.Object newSourceObject)
  {
    org.xtext.textualSlco.Object oldSourceObject = sourceObject;
    sourceObject = newSourceObject;
    if (eNotificationRequired())
      eNotify(new ENotificationImpl(this, Notification.SET, TextualSlcoPackage.UNIDIRECTIONAL_CHANNEL__SOURCE_OBJECT, oldSourceObject, sourceObject));
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public PortReference getSourcePort()
  {
    return sourcePort;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public NotificationChain basicSetSourcePort(PortReference newSourcePort, NotificationChain msgs)
  {
    PortReference oldSourcePort = sourcePort;
    sourcePort = newSourcePort;
    if (eNotificationRequired())
    {
      ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, TextualSlcoPackage.UNIDIRECTIONAL_CHANNEL__SOURCE_PORT, oldSourcePort, newSourcePort);
      if (msgs == null) msgs = notification; else msgs.add(notification);
    }
    return msgs;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public void setSourcePort(PortReference newSourcePort)
  {
    if (newSourcePort != sourcePort)
    {
      NotificationChain msgs = null;
      if (sourcePort != null)
        msgs = ((InternalEObject)sourcePort).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - TextualSlcoPackage.UNIDIRECTIONAL_CHANNEL__SOURCE_PORT, null, msgs);
      if (newSourcePort != null)
        msgs = ((InternalEObject)newSourcePort).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - TextualSlcoPackage.UNIDIRECTIONAL_CHANNEL__SOURCE_PORT, null, msgs);
      msgs = basicSetSourcePort(newSourcePort, msgs);
      if (msgs != null) msgs.dispatch();
    }
    else if (eNotificationRequired())
      eNotify(new ENotificationImpl(this, Notification.SET, TextualSlcoPackage.UNIDIRECTIONAL_CHANNEL__SOURCE_PORT, newSourcePort, newSourcePort));
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public org.xtext.textualSlco.Object getTargetObject()
  {
    if (targetObject != null && targetObject.eIsProxy())
    {
      InternalEObject oldTargetObject = (InternalEObject)targetObject;
      targetObject = (org.xtext.textualSlco.Object)eResolveProxy(oldTargetObject);
      if (targetObject != oldTargetObject)
      {
        if (eNotificationRequired())
          eNotify(new ENotificationImpl(this, Notification.RESOLVE, TextualSlcoPackage.UNIDIRECTIONAL_CHANNEL__TARGET_OBJECT, oldTargetObject, targetObject));
      }
    }
    return targetObject;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public org.xtext.textualSlco.Object basicGetTargetObject()
  {
    return targetObject;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public void setTargetObject(org.xtext.textualSlco.Object newTargetObject)
  {
    org.xtext.textualSlco.Object oldTargetObject = targetObject;
    targetObject = newTargetObject;
    if (eNotificationRequired())
      eNotify(new ENotificationImpl(this, Notification.SET, TextualSlcoPackage.UNIDIRECTIONAL_CHANNEL__TARGET_OBJECT, oldTargetObject, targetObject));
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public PortReference getTargetPort()
  {
    return targetPort;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public NotificationChain basicSetTargetPort(PortReference newTargetPort, NotificationChain msgs)
  {
    PortReference oldTargetPort = targetPort;
    targetPort = newTargetPort;
    if (eNotificationRequired())
    {
      ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, TextualSlcoPackage.UNIDIRECTIONAL_CHANNEL__TARGET_PORT, oldTargetPort, newTargetPort);
      if (msgs == null) msgs = notification; else msgs.add(notification);
    }
    return msgs;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public void setTargetPort(PortReference newTargetPort)
  {
    if (newTargetPort != targetPort)
    {
      NotificationChain msgs = null;
      if (targetPort != null)
        msgs = ((InternalEObject)targetPort).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - TextualSlcoPackage.UNIDIRECTIONAL_CHANNEL__TARGET_PORT, null, msgs);
      if (newTargetPort != null)
        msgs = ((InternalEObject)newTargetPort).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - TextualSlcoPackage.UNIDIRECTIONAL_CHANNEL__TARGET_PORT, null, msgs);
      msgs = basicSetTargetPort(newTargetPort, msgs);
      if (msgs != null) msgs.dispatch();
    }
    else if (eNotificationRequired())
      eNotify(new ENotificationImpl(this, Notification.SET, TextualSlcoPackage.UNIDIRECTIONAL_CHANNEL__TARGET_PORT, newTargetPort, newTargetPort));
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  @Override
  public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID, NotificationChain msgs)
  {
    switch (featureID)
    {
      case TextualSlcoPackage.UNIDIRECTIONAL_CHANNEL__SOURCE_PORT:
        return basicSetSourcePort(null, msgs);
      case TextualSlcoPackage.UNIDIRECTIONAL_CHANNEL__TARGET_PORT:
        return basicSetTargetPort(null, msgs);
    }
    return super.eInverseRemove(otherEnd, featureID, msgs);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  @Override
  public Object eGet(int featureID, boolean resolve, boolean coreType)
  {
    switch (featureID)
    {
      case TextualSlcoPackage.UNIDIRECTIONAL_CHANNEL__SOURCE_OBJECT:
        if (resolve) return getSourceObject();
        return basicGetSourceObject();
      case TextualSlcoPackage.UNIDIRECTIONAL_CHANNEL__SOURCE_PORT:
        return getSourcePort();
      case TextualSlcoPackage.UNIDIRECTIONAL_CHANNEL__TARGET_OBJECT:
        if (resolve) return getTargetObject();
        return basicGetTargetObject();
      case TextualSlcoPackage.UNIDIRECTIONAL_CHANNEL__TARGET_PORT:
        return getTargetPort();
    }
    return super.eGet(featureID, resolve, coreType);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  @Override
  public void eSet(int featureID, Object newValue)
  {
    switch (featureID)
    {
      case TextualSlcoPackage.UNIDIRECTIONAL_CHANNEL__SOURCE_OBJECT:
        setSourceObject((org.xtext.textualSlco.Object)newValue);
        return;
      case TextualSlcoPackage.UNIDIRECTIONAL_CHANNEL__SOURCE_PORT:
        setSourcePort((PortReference)newValue);
        return;
      case TextualSlcoPackage.UNIDIRECTIONAL_CHANNEL__TARGET_OBJECT:
        setTargetObject((org.xtext.textualSlco.Object)newValue);
        return;
      case TextualSlcoPackage.UNIDIRECTIONAL_CHANNEL__TARGET_PORT:
        setTargetPort((PortReference)newValue);
        return;
    }
    super.eSet(featureID, newValue);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  @Override
  public void eUnset(int featureID)
  {
    switch (featureID)
    {
      case TextualSlcoPackage.UNIDIRECTIONAL_CHANNEL__SOURCE_OBJECT:
        setSourceObject((org.xtext.textualSlco.Object)null);
        return;
      case TextualSlcoPackage.UNIDIRECTIONAL_CHANNEL__SOURCE_PORT:
        setSourcePort((PortReference)null);
        return;
      case TextualSlcoPackage.UNIDIRECTIONAL_CHANNEL__TARGET_OBJECT:
        setTargetObject((org.xtext.textualSlco.Object)null);
        return;
      case TextualSlcoPackage.UNIDIRECTIONAL_CHANNEL__TARGET_PORT:
        setTargetPort((PortReference)null);
        return;
    }
    super.eUnset(featureID);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  @Override
  public boolean eIsSet(int featureID)
  {
    switch (featureID)
    {
      case TextualSlcoPackage.UNIDIRECTIONAL_CHANNEL__SOURCE_OBJECT:
        return sourceObject != null;
      case TextualSlcoPackage.UNIDIRECTIONAL_CHANNEL__SOURCE_PORT:
        return sourcePort != null;
      case TextualSlcoPackage.UNIDIRECTIONAL_CHANNEL__TARGET_OBJECT:
        return targetObject != null;
      case TextualSlcoPackage.UNIDIRECTIONAL_CHANNEL__TARGET_PORT:
        return targetPort != null;
    }
    return super.eIsSet(featureID);
  }

} //UnidirectionalChannelImpl
